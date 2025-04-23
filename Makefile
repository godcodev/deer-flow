.PHONY: lint format install-dev serve test coverage

install-dev:
	uv pip install -e ".[dev]" && uv pip install -e ".[test]"

format:
	uv run black --preview .

lint:
	uv run black --check .

serve:
	uv run server.py --reload

test:
	uv run pytest tests/

coverage:
	uv run pytest --cov=src tests/ --cov-report=term-missing
