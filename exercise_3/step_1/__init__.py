"""Exercise 3.1 package."""

try:
    from .workflow import run_training_workflow
except ImportError:  # pragma: no cover - inference-only environments
    run_training_workflow = None

__all__ = ["run_training_workflow"]
