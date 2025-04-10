import os
import enum
from dotenv import load_dotenv

load_dotenv()


class SearchEngine(enum.Enum):
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"


# Tool configuration
SELECTED_SEARCH_ENGINE = os.getenv("SEARCH_API", SearchEngine.TAVILY.value)
SEARCH_MAX_RESULTS = 3
