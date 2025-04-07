from .tools import TAVILY_MAX_RESULTS
from .loader import load_yaml_config

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Team configuration
TEAM_MEMBER_CONFIGRATIONS = {
    "researcher": {
        "name": "researcher",
        "desc": (
            "Responsible for searching and collecting relevant information, understanding user needs and conducting research analysis"
        ),
        "desc_for_llm": (
            "Uses search engines and web crawlers to gather information from the internet. "
            "Outputs a Markdown report summarizing findings. Researcher can not do math or programming."
        ),
        "is_optional": False,
    },
    "coder": {
        "name": "coder",
        "desc": (
            "Responsible for code implementation, debugging and optimization, handling technical programming tasks"
        ),
        "desc_for_llm": (
            "Executes Python or Bash commands, performs mathematical calculations, and outputs a Markdown report. "
            "Must be used for all mathematical computations."
        ),
        "is_optional": True,
    },
}

TEAM_MEMBERS = list(TEAM_MEMBER_CONFIGRATIONS.keys())

__all__ = [
    # Other configurations
    "TEAM_MEMBERS",
    "TEAM_MEMBER_CONFIGRATIONS",
    "TAVILY_MAX_RESULTS",
]
