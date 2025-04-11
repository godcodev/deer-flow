import logging
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import BraveSearch
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper, BraveSearchWrapper
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

LoggedBraveSearch = create_logged_tool(BraveSearch)
brave_search_tool = LoggedBraveSearch(
    name="web_search",
    search_wrapper=BraveSearchWrapper(
        api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
        search_kwargs={"count": SEARCH_MAX_RESULTS},
    ),
)

LoggedArxivSearch = create_logged_tool(ArxivQueryRun)
arxiv_search_tool = LoggedArxivSearch(
    name="web_search",
    api_wrapper=ArxivAPIWrapper(
        top_k_results=SEARCH_MAX_RESULTS,
        load_max_docs=SEARCH_MAX_RESULTS,
        load_all_available_meta=True,
    ),
)
