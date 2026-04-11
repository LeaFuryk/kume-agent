from kume.infrastructure.config import Settings
from kume.infrastructure.logging import setup_logging
from kume.infrastructure.metrics import MetricsCallbackHandler, MetricsCollector

__all__ = [
    "Container",
    "MetricsCallbackHandler",
    "MetricsCollector",
    "Settings",
    "setup_logging",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "Container":
        from kume.infrastructure.container import Container

        return Container
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
