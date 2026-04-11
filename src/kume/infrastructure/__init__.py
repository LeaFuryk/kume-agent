from kume.infrastructure.config import Settings
from kume.infrastructure.logging import setup_logging

__all__ = [
    "Container",
    "MetricsCallbackHandler",
    "MetricsCollector",
    "Settings",
    "setup_logging",
]

_LAZY_IMPORTS = {
    "Container": "kume.infrastructure.container",
    "MetricsCollector": "kume.infrastructure.metrics",
    "MetricsCallbackHandler": "kume.infrastructure.metrics",
}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    module_path = _LAZY_IMPORTS.get(name)
    if module_path:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
