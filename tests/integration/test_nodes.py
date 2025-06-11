import json
import pytest
from unittest.mock import patch, MagicMock

# 在这里 mock 掉 get_llm_by_type，避免 ValueError
with patch("src.llms.llm.get_llm_by_type", return_value=MagicMock()):
    from langgraph.types import Command
    from src.graph.nodes import background_investigation_node
    from src.config import SearchEngine
    from langchain_core.messages import HumanMessage

# Mock data
MOCK_SEARCH_RESULTS = [
    {"title": "Test Title 1", "content": "Test Content 1"},
    {"title": "Test Title 2", "content": "Test Content 2"},
]


@pytest.fixture
def mock_state():
    return {
        "messages": [HumanMessage(content="test query")],
        "research_topic": "test query",
        "background_investigation_results": None,
    }


@pytest.fixture
def mock_configurable():
    mock = MagicMock()
    mock.max_search_results = 5
    return mock


@pytest.fixture
def mock_config():
    # 你可以根据实际需要返回一个 MagicMock 或 dict
    return MagicMock()


@pytest.fixture
def patch_config_from_runnable_config(mock_configurable):
    with patch(
        "src.graph.nodes.Configuration.from_runnable_config",
        return_value=mock_configurable,
    ):
        yield


@pytest.fixture
def mock_tavily_search():
    with patch("src.graph.nodes.LoggedTavilySearch") as mock:
        instance = mock.return_value
        instance.invoke.return_value = [
            {"title": "Test Title 1", "content": "Test Content 1"},
            {"title": "Test Title 2", "content": "Test Content 2"},
        ]
        yield mock


@pytest.fixture
def mock_web_search_tool():
    with patch("src.graph.nodes.get_web_search_tool") as mock:
        instance = mock.return_value
        instance.invoke.return_value = [
            {"title": "Test Title 1", "content": "Test Content 1"},
            {"title": "Test Title 2", "content": "Test Content 2"},
        ]
        yield mock


@pytest.mark.parametrize("search_engine", [SearchEngine.TAVILY.value, "other"])
def test_background_investigation_node_tavily(
    mock_state,
    mock_tavily_search,
    mock_web_search_tool,
    search_engine,
    patch_config_from_runnable_config,
    mock_config,
):
    """Test background_investigation_node with Tavily search engine"""
    with patch("src.graph.nodes.SELECTED_SEARCH_ENGINE", search_engine):
        result = background_investigation_node(mock_state, mock_config)

        # Verify the result structure
        assert isinstance(result, dict)

        # Verify the update contains background_investigation_results
        assert "background_investigation_results" in result

        # Parse and verify the JSON content
        results = result["background_investigation_results"]

        if search_engine == SearchEngine.TAVILY.value:
            mock_tavily_search.return_value.invoke.assert_called_once_with("test query")
            assert (
                results
                == "## Test Title 1\n\nTest Content 1\n\n## Test Title 2\n\nTest Content 2"
            )
        else:
            mock_web_search_tool.return_value.invoke.assert_called_once_with(
                "test query"
            )
            assert len(json.loads(results)) == 2


def test_background_investigation_node_malformed_response(
    mock_state, mock_tavily_search, patch_config_from_runnable_config, mock_config
):
    """Test background_investigation_node with malformed Tavily response"""
    with patch("src.graph.nodes.SELECTED_SEARCH_ENGINE", SearchEngine.TAVILY.value):
        # Mock a malformed response
        mock_tavily_search.return_value.invoke.return_value = "invalid response"

        result = background_investigation_node(mock_state, mock_config)

        # Verify the result structure
        assert isinstance(result, dict)

        # Verify the update contains background_investigation_results
        assert "background_investigation_results" in result

        # Parse and verify the JSON content
        results = result["background_investigation_results"]
        assert json.loads(results) is None
