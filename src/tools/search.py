import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from src.config import SEARCH_MAX_RESULTS
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)


LoggedTavilySearch = create_logged_tool(TavilySearchResults)
tavily_search_tool = LoggedTavilySearch(
    name="web_search", max_results=SEARCH_MAX_RESULTS
)

LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
duckduckgo_search_tool = LoggedDuckDuckGoSearch(
    name="web_search", max_results=SEARCH_MAX_RESULTS
)
