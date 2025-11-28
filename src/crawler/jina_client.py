import logging
import os
import requests

logger = logging.getLogger(__name__)


class JinaClient:
    def __init__(self):
        # Cache the key once instead of calling getenv twice later
        self.api_key = os.getenv("JINA_API_KEY")

        # Reuse a session for performance (same behavior for a single request)
        self.session = requests.Session()

    def crawl(self, url: str, return_format: str = "html") -> str:
        headers = {
            "Content-Type": "application/json",
            "X-Return-Format": return_format,
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            logger.warning(
                "Jina API key is not set. Provide your own key to access a higher rate limit. "
                "See https://jina.ai/reader for more information."
            )

        response = self.session.post(
            "https://r.jina.ai/",
            headers=headers,
            json={"url": url}
        )

        return response.text
