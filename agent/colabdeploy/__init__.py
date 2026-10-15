"""ColabDeploy agent - the disposable compute plane.

One cell in a Colab notebook is the whole user experience::

    !pip -q install git+https://github.com/Nileshsri2022/deploy-ml-models-for-free-colab-ui#subdirectory=agent
    from colabdeploy import launch
    launch(gateway_url="https://your-gateway.example.com", token="CDP_RT_...")

``launch`` is idempotent: re-running the cell (or resuming a session) re-attaches
without duplicating servers or tunnels, and re-hydrates models from the on-disk
cache so a resume takes seconds instead of minutes.
"""

from colabdeploy.launch import launch, launch_in_thread, status

__version__ = "0.1.0"

__all__ = ["launch", "launch_in_thread", "status", "__version__"]
