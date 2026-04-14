import os

import pytest


# Auto-skip eval tests when no API key is available
def pytest_collection_modifyitems(config, items):
    if not os.environ.get("OPENAI_API_KEY"):
        skip_marker = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            if "eval" in item.keywords:
                item.add_marker(skip_marker)
