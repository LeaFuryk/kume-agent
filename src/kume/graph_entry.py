"""Entry point for LangGraph Platform deployment."""

from kume.infrastructure.config import Settings
from kume.infrastructure.container import Container

settings = Settings.from_env()
container = Container(settings)
graph = container.build_graph()
