"""QR Code Monster (ControlNet + SD 1.5) as a deployable ColabDeploy model.

This file exists so that a model like ``monster-labs/control_v1p_sd15_qrcode_monster``
can be added **entirely from the control-plane UI** — no notebook editing, no
``%%writefile``, no code to maintain. The UI's "Register by URI" form accepts an
HTTPS URL, the agent downloads this file onto the runtime VM, and calls the
``predict`` function below for every request.

Register it like this (Models -> Register by URI):

    Artifact URI : https://raw.githubusercontent.com/<owner>/<repo>/<branch>/examples/qr_monster_predict.py
    Name         : qrmonster
    Framework    : auto
    Adapter      : python_function
    Entrypoint   : predict
    Input schema : {"fields": {"prompt": {"type": "string"}, "qr_text": {"type": "string"},
                               "steps": {"type": "integer"}, "scale": {"type": "number"},
                               "size": {"type": "integer"}, "seed": {"type": "integer"}},
                    "required": ["prompt"],
                    "example": {"prompt": "a lush jungle with a hidden monster",
                                "qr_text": "https://qrcode.monster"}}
    Warm-up      : {"prompt": "warmup", "qr_text": "warmup", "steps": 2, "size": 512}

Requirements on the runtime VM (one cell in Colab, plus a GPU runtime):

    !pip -q install -U diffusers transformers accelerate safetensors qrcode pillow
    !pip -q uninstall -y torchao     # see the torchao note below

And on the gateway, because a diffusion step is slower than the 30 s default:

    CDP_TIMEOUT_MS=120000      # Render -> Environment

Colab pitfall (torchao): Colab preinstalls an old ``torchao`` (0.14.x) while current
diffusers imports ``torchao.quantization.FqnToConfig`` unconditionally, and that name
only exists from torchao 0.15.0. The result is

    cannot import name 'FqnToConfig' from 'torchao.quantization'

which kills ``import diffusers`` before this file runs a single line. Removing torchao
is the safest fix (diffusers only needs it for quantization, which nothing here uses);
upgrading it to >= 0.15 also works. The error message below says the same thing if you
hit it later.

Why ``python_function`` and not the ``huggingface`` adapter? That adapter builds a
``transformers.pipeline()`` from a repo, which cannot express "ControlNet + a base
checkpoint + a QR image as the conditioning input". Anything that is a diffusers
*pipeline* with custom preprocessing needs an entrypoint like this one.

The QR image handling follows the model card: 16 px modules, a #808080 background
(it blends into the generated image better than white), and a high error-correction
level so the code still reads after the model has painted over it.
"""

from __future__ import annotations

import base64
import io
import os
import random
from typing import Any

#: ControlNet weights. v2 lives in the ``v2/`` subfolder and is a big upgrade over
#: v1 for both scannability and creativity (per the model card).
CONTROLNET_REPO = os.getenv("QR_CONTROLNET_REPO", "monster-labs/control_v1p_sd15_qrcode_monster")
CONTROLNET_SUBFOLDER = os.getenv("QR_CONTROLNET_SUBFOLDER", "v2")

#: Base checkpoint. ``runwayml/stable-diffusion-v1-5`` was taken down in Aug 2024;
#: this is the community mirror of the same weights.
BASE_REPO = os.getenv("QR_BASE_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5")

#: Module size and background per the model card. ``#808080`` is the documented trick.
QR_MODULE_PX = 16
QR_BG = "#808080"

_PIPE: Any = None
_PIPE_INFO: dict[str, Any] = {}

_MISSING_DEPS = (
    "this model needs diffusers/transformers/torch and qrcode on the runtime VM. "
    "In Colab run:  !pip -q install -U diffusers transformers accelerate safetensors qrcode pillow"
)


# --------------------------------------------------------------------------- #
# Heavy imports stay inside functions: importing this file must be cheap, and a
# missing framework must surface as one clear message at load time instead of an
# import error nobody can place.
# --------------------------------------------------------------------------- #
def _load() -> Any:
    """Build the ControlNet pipeline once, on first use."""
    global _PIPE, _PIPE_INFO
    try:
        import torch  # type: ignore
        import qrcode  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"missing dependency '{exc.name}': {_MISSING_DEPS}") from exc

    try:
        from diffusers import (  # type: ignore
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )
    except ImportError as exc:
        # The single most common Colab failure: Colab ships an old ``torchao``
        # (0.14.x) while current diffusers imports ``torchao.quantization.FqnToConfig``
        # unconditionally - that name only exists from torchao 0.15.0 on. diffusers
        # then refuses to import the ControlNet module at all. Nothing to do with
        # this file or the model - it is the environment's pairing.
        if "FqnToConfig" in str(exc) or "torchao" in str(exc):
            raise RuntimeError(
                "this VM's torchao is too old for the installed diffusers "
                "(diffusers wants torchao.quantization.FqnToConfig, added in torchao 0.15). "
                "Fix it in Colab and re-attach:\n"
                "    !pip -q uninstall -y torchao        # recommended: diffusers works without it\n"
                "    !pip -q install -U torchao          # or upgrade it to >= 0.15\n"
                "then run the launch cell again (or Runtimes -> Reconcile in the UI)."
            ) from exc
        raise RuntimeError(f"missing dependency '{exc.name}': {_MISSING_DEPS}") from exc

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_REPO, subfolder=CONTROLNET_SUBFOLDER, torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_REPO, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # Cheap on a T4, and the difference between "fits" and OOM at 768 px.
    pipe.enable_attention_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    _PIPE = pipe
    _PIPE_INFO = {
        "base": BASE_REPO,
        "controlnet": f"{CONTROLNET_REPO}/{CONTROLNET_SUBFOLDER}",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
    }
    return pipe


def _pipe() -> Any:
    return _PIPE if _PIPE is not None else _load()


def _qr_image(text: str, size: int) -> Any:
    """QR code as the conditioning image: gray background, 16 px modules, square."""
    import qrcode  # type: ignore
    from PIL import Image  # type: ignore

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=QR_MODULE_PX,
        border=2,
    )
    qr.add_data(text or " ")
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color=QR_BG).convert("RGB")
    return image.resize((size, size), Image.NEAREST)


def _png_b64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def predict(payload: dict[str, Any], model: Any = None, meta: Any = None) -> dict[str, Any]:
    """Generate a scannable-art QR image. ``payload`` is the request's ``inputs``."""
    payload = payload or {}
    prompt = str(payload.get("prompt") or payload.get("text") or "").strip()
    if not prompt:
        raise ValueError("'prompt' is required, e.g. {\"prompt\": \"a blue monster holding a sign\"}")

    # The QR payload defaults to the prompt so a single-field request still works.
    qr_text = str(payload.get("qr_text") or payload.get("url") or prompt)
    negative = str(payload.get("negative_prompt") or "ugly, blurry, low quality, watermark")
    steps = int(_clamp(payload.get("steps"), 1, 40, 20))
    scale = _clamp(payload.get("scale"), 0.0, 2.0, 1.3)
    # 512 keeps the payload small; 768 is sharper but ~2x slower.
    size = 768 if int(_clamp(payload.get("size"), 512, 768, 512)) >= 640 else 512

    pipe = _pipe()
    control_image = _qr_image(qr_text, size)

    import torch  # type: ignore

    device = _PIPE_INFO.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed_raw = payload.get("seed")
    seed = int(seed_raw) if str(seed_raw or "").strip() not in ("", "None") else random.randint(0, 2**31 - 1)
    generator = torch.Generator(device if device == "cuda" else "cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=control_image,
        num_inference_steps=steps,
        guidance_scale=float(_clamp(payload.get("guidance"), 1.0, 20.0, 7.5)),
        controlnet_conditioning_scale=scale,
        generator=generator,
        width=size,
        height=size,
    )
    image = result.images[0]

    return {
        "image_b64": _png_b64(image),
        "mime": "image/png",
        "width": size,
        "height": size,
        "seed": seed,
        "steps": steps,
        "controlnet_scale": scale,
        "prompt": prompt,
        "qr_text": qr_text,
        # The conditioning image, so a caller can eyeball what the model was told.
        "qr_b64": _png_b64(control_image),
        "model": _PIPE_INFO,
        "note": "decode image_b64 to a .png; not every seed scans - vary seed/scale/prompt",
    }


def status() -> dict[str, Any]:
    """Handy when debugging from the Playground or a shell."""
    return {"loaded": _PIPE is not None, **_PIPE_INFO}
