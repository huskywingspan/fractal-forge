"""FractalForge interactive viewer -- real-time fractal exploration with Dear PyGui."""


def launch_viewer():
    """Launch the interactive fractal viewer.

    Requires dearpygui to be installed:
        pip install dearpygui
    """
    try:
        import dearpygui.dearpygui  # noqa: F401
    except ImportError:
        raise ImportError(
            "Dear PyGui is required for the viewer. Install it with:\n"
            "  pip install dearpygui"
        )

    from fractalforge.viewer.app import ViewerApp

    app = ViewerApp()
    app.run()
