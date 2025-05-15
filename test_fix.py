#!/usr/bin/env python3
"""
This script manually patches sys.modules to fix the LLM import issue
so that tests can run without requiring LLM configuration.
"""

import sys
from unittest.mock import MagicMock

# Create mocks
mock_llm = MagicMock()
mock_llm.invoke.return_value = "Mock LLM response"

# Create a mock module for llm.py
mock_llm_module = MagicMock()
mock_llm_module.get_llm_by_type = lambda llm_type: mock_llm
mock_llm_module.basic_llm = mock_llm
mock_llm_module._create_llm_use_conf = lambda llm_type, conf: mock_llm

# Set the mock module
sys.modules["src.llms.llm"] = mock_llm_module

print("Successfully patched LLM module. You can now run your tests.")
print("Example: uv run pytest tests/test_types.py -v")
