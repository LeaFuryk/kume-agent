"""Entry point for LangGraph Platform deployment.

This module is referenced by langgraph.json and exposes the compiled graph.

Limitations of direct graph invocation (vs Telegram bot):
- Image analysis (AnalyzeFoodImageTool) is not available -- image upload
  requires the Telegram adapter which handles file downloads and staging.
- Audio transcription (Whisper) is similarly unavailable.
- These multimodal features only work via the Telegram bot path.
"""

import logging

from kume.infrastructure.config import Settings
from kume.infrastructure.container import Container

try:
    settings = Settings.from_env()
    container = Container(settings)
    graph = container.build_graph()
except Exception as e:
    logging.getLogger("kume.graph_entry").error("Failed to build graph: %s", e, exc_info=True)
    raise
