import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def resolve_model_cache_path(model_cache_path: str | Path | None):
    """Context manager that yields a concrete Path for model_cache_path.

    - If model_cache_path is None, creates a TemporaryDirectory and yields its Path.
      The directory is cleaned up on context exit.
    - Otherwise, yields Path(model_cache_path) without creating or cleaning.
    """
    if model_cache_path is None:
        with tempfile.TemporaryDirectory(prefix="sae_probes_model_cache_") as td:
            yield Path(td)
    else:
        yield Path(model_cache_path)
