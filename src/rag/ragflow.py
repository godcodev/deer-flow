import os
from typing import List, Optional
from urllib.parse import urlparse

import requests

from src.rag.retriever import Chunk, Document, Resource, Retriever


class RAGFlowProvider(Retriever):
    """
    RAGFlowProvider is a provider that uses RAGFlow to retrieve documents.
    """

    api_url: str
    api_key: str
    page_size: int = 10
    cross_languages: Optional[List[str]] = None

    def __init__(self):
        def get_env(name: str, required: bool = True) -> Optional[str]:
            value = os.getenv(name)
            if required and not value:
                raise ValueError(f"{name} is not set")
            return value

        self.api_url = get_env("RAGFLOW_API_URL")
        self.api_key = get_env("RAGFLOW_API_KEY")

        page_size = os.getenv("RAGFLOW_PAGE_SIZE")
        if page_size:
            self.page_size = int(page_size)

        cross_lang = os.getenv("RAGFLOW_CROSS_LANGUAGES")
        self.cross_languages = cross_lang.split(",") if cross_lang else None

    def query_relevant_documents(
        self, query: str, resources: Optional[list[Resource]] = None
    ) -> list[Document]:
        resources = resources or []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Extract dataset_ids and document_ids in a single pass
        dataset_ids = []
        document_ids = []

        for resource in resources:
            dataset_id, document_id = parse_uri(resource.uri)
            dataset_ids.append(dataset_id)
            if document_id:
                document_ids.append(document_id)

        payload = {
            "question": query,
            "dataset_ids": dataset_ids,
            "document_ids": document_ids,
            "page_size": self.page_size,
        }

        if self.cross_languages:
            payload["cross_languages"] = self.cross_languages

        response = requests.post(
            f"{self.api_url}/api/v1/retrieval",
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to query documents: {response.text}")

        data = response.json().get("data", {})
        doc_aggs = data.get("doc_aggs", [])

        # Create document dictionary directly using a dict comprehension
        docs = {
            d["doc_id"]: Document(
                id=d["doc_id"],
                title=d.get("doc_name"),
                chunks=[]
            )
            for d in doc_aggs
            if d.get("doc_id")
        }

        # Append chunks to matching documents
        for chunk in data.get("chunks", []):
            doc_id = chunk.get("document_id")
            if doc_id in docs:
                docs[doc_id].chunks.append(
                    Chunk(
                        content=chunk.get("content"),
                        similarity=chunk.get("similarity"),
                    )
                )

        return list(docs.values())


    def list_resources(self, query: str | None = None) -> list[Resource]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        params = {"name": query} if query else {}

        response = requests.get(
            f"{self.api_url}/api/v1/datasets",
            headers=headers,
            params=params,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to list resources: {response.text}")

        items = response.json().get("data", [])
        return [
            Resource(
                uri=f"rag://dataset/{item.get('id')}",
                title=item.get("name", ""),
                description=item.get("description", ""),
            )
            for item in items
        ]


def parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "rag":
        raise ValueError(f"Invalid URI: {uri}")
    return parsed.path.split("/")[1], parsed.fragment
