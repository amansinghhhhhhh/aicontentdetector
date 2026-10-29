"""AI content detection ML package."""

__all__ = ["analyze_text", "get_model", "warmup"]


def __getattr__(name):
    if name in ("analyze_text", "get_model", "warmup"):
        from ml import detect

        return getattr(detect, name)
    raise AttributeError(name)
