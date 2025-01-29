"""Set the version of the package."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    try:
        from importlib_metadata import PackageNotFoundError, version  # type: ignore
    except ImportError:
        raise ImportError("You must have the `importlib_metadata` package installed")

try:
    __version__ = version("bio_curve_fit")
except PackageNotFoundError:
    __version__ = "0.0.0"
