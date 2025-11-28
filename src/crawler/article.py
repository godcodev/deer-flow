import re
from urllib.parse import urljoin
from markdownify import markdownify as md


class Article:
    url: str

    def __init__(self, title: str, html_content: str):
        self.title = title
        self.html_content = html_content

    def to_markdown(self, including_title: bool = True) -> str:
        if including_title:
            return f"# {self.title}\n\n{md(self.html_content)}"
        return md(self.html_content)

    def to_message(self) -> list[dict]:
        image_pattern = r"!\[.*?\]\((.*?)\)"

        markdown = self.to_markdown()
        parts = re.split(image_pattern, markdown)

        content = []
        for i, part in enumerate(parts):
            part = part.strip()
            if i % 2 == 1:  # image URLs
                content.append({
                    "type": "image_url",
                    "image_url": {"url": urljoin(self.url, part)}
                })
            else:  # text blocks
                if part:  # avoid adding empty text blocks
                    content.append({"type": "text", "text": part})

        return content
