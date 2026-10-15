"""Serving the UI from the control-plane process.

One process serving both the API and the SPA is the recommended free-tier
topology, which makes these three behaviours load-bearing rather than cosmetic:

* a browser at ``/`` gets the app, a script gets JSON metadata (content
  negotiation, so neither audience is broken);
* deep links and pasted URLs fall back to ``index.html`` instead of 404-ing;
* a crafted path cannot escape the build directory and read the host.

The tests build a fake ``dist/`` so they run whether or not the UI has been built
locally (``apps/ui/dist`` is gitignored and built in a separate CI job).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head><title>ColabDeploy</title></head>
  <body>
    <div id="root"></div>
    <script type="module" src="./assets/app.js"></script>
  </body>
</html>
"""


@pytest.fixture()
def dist(tmp_path: Path) -> Path:
    """A minimal build output: index.html, one asset, and a file outside it."""
    build = tmp_path / "dist"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (build / "assets" / "app.js").write_text("console.log('colabdeploy');", encoding="utf-8")
    (build / "assets" / "app.css").write_text(":root{--bg:#0b0d10}", encoding="utf-8")
    # Must never be reachable through /ui/{path}
    (tmp_path / "secret.txt").write_text("CDP_ADMIN_KEY=leaked", encoding="utf-8")
    return build


@pytest.fixture()
def ui_client(state: Any, dist: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A TestClient for an app whose SPA lives in the temp build directory."""
    from fastapi.testclient import TestClient

    from gateway.main import create_app

    monkeypatch.setenv("CDP_UI_DIST", str(dist))
    app = create_app(state)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def bare_client(state: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The same app with no build present - what a fresh checkout looks like."""
    from fastapi.testclient import TestClient

    from gateway.main import create_app

    monkeypatch.setenv("CDP_UI_DIST", str(tmp_path / "never-built"))
    app = create_app(state)
    with TestClient(app) as client:
        yield client


# --------------------------------------------------------------------------- #
# Content negotiation at the site root
# --------------------------------------------------------------------------- #
def test_browser_at_root_gets_the_app(ui_client: Any) -> None:
    response = ui_client.get("/", headers={"Accept": BROWSER_ACCEPT, "User-Agent": BROWSER_UA})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="root">' in response.text


def test_curl_at_root_gets_json_metadata(ui_client: Any) -> None:
    """``Accept: */*`` with no browser UA is a script, not a person."""
    response = ui_client.get("/", headers={"Accept": "*/*", "User-Agent": "curl/8.5.0"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["name"] == "ColabDeploy"
    assert body["public_api"] == "/v1"
    assert body["management_api"] == "/api"
    assert body["agent_api"] == "/internal"


def test_default_testclient_accept_is_treated_as_a_script(ui_client: Any) -> None:
    """The existing ``GET /`` contract (JSON) must not regress for API callers."""
    response = ui_client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "ColabDeploy"


def test_webview_with_wildcard_accept_and_browser_ua_gets_the_app(ui_client: Any) -> None:
    """Some embedded browsers send only ``*/*`` plus a Mozilla UA."""
    response = ui_client.get("/", headers={"Accept": "*/*", "User-Agent": BROWSER_UA})
    assert response.headers["content-type"].startswith("text/html")


def test_root_reports_the_ui_is_available(ui_client: Any) -> None:
    assert ui_client.get("/").json()["ui"] == "/ui"


# --------------------------------------------------------------------------- #
# The /ui mount
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ["/ui", "/ui/", "/ui/index.html"])
def test_ui_root_serves_the_app(ui_client: Any, path: str) -> None:
    response = ui_client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="root">' in response.text


@pytest.mark.parametrize("path", ["/ui/models", "/ui/cache/entries", "/ui/deep/nested/route"])
def test_ui_deep_links_fall_back_to_index(ui_client: Any, path: str) -> None:
    """A pasted link must land on the app, not an error page."""
    response = ui_client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_ui_assets_are_served_with_the_right_type(ui_client: Any) -> None:
    js = ui_client.get("/ui/assets/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "colabdeploy" in js.text

    css = ui_client.get("/ui/assets/app.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]


def test_root_assets_mount_matches_the_relative_bundle_urls(ui_client: Any) -> None:
    """The bundle is built with ``base: './'`` and served from ``/`` too."""
    response = ui_client.get("/assets/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_unknown_asset_is_a_404_not_index_html(ui_client: Any) -> None:
    """Returning HTML for a missing .js would produce a confusing MIME error."""
    response = ui_client.get("/ui/assets/does-not-exist.js")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        "/ui/%2e%2e/secret.txt",
        "/ui/..%2fsecret.txt",
        "/ui/assets/../../secret.txt",
    ],
)
def test_encoded_traversal_cannot_escape_the_build_dir(ui_client: Any, path: str) -> None:
    response = ui_client.get(path)
    assert "leaked" not in response.text
    # Either the SPA fallback or a 404 - never the file above the build dir.
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.headers["content-type"].startswith("text/html")


def test_absolute_host_file_is_not_served(ui_client: Any, tmp_path: Path) -> None:
    response = ui_client.get(f"/ui/{tmp_path / 'secret.txt'}")
    assert "leaked" not in response.text


# --------------------------------------------------------------------------- #
# No build present
# --------------------------------------------------------------------------- #
def test_without_a_build_root_is_json_and_ui_is_absent(bare_client: Any) -> None:
    root = bare_client.get("/", headers={"Accept": BROWSER_ACCEPT, "User-Agent": BROWSER_UA})
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("application/json")
    assert root.json()["ui"] is None


def test_without_a_build_the_api_still_works(bare_client: Any) -> None:
    assert bare_client.get("/healthz").status_code == 200
    assert bare_client.get("/v1/status").status_code == 200


def test_favicon_is_never_a_500(ui_client: Any) -> None:
    """Browsers request it unconditionally; a missing one must not pollute logs."""
    response = ui_client.get("/favicon.ico")
    assert response.status_code in (200, 404)
