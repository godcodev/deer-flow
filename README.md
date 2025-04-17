# 🦌 DeerFlow

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Come from Open Source, Back to Open Source

**DeerFlow** (**D**eep **E**xploration and **E**fficient **R**esearch **Flow**) is a community-driven AI automation framework that builds upon the incredible work of the open source community. Our goal is to combine language models with specialized tools for tasks like web search, crawling, and Python code execution, while giving back to the community that made this possible.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow

# Install dependencies, uv will take care of the python interpreter and venv creation, and install the required packages
uv sync

# Configure .env with your Search Engine API keys
# Tavily: https://app.tavily.com/home
# Brave_SEARCH: https://brave.com/search/api/
cp .env.example .env

# See the 'Supported Search Engines' section below for all available options

# Configure conf.yaml for your LLM model and API keys
# Gemini: https://ai.google.dev/gemini-api/docs/openai
cp conf.yaml.example conf.yaml

# Run the project
uv run main.py
```

## Web UI

This project also includes a web UI that allows you to interact with the deep researcher.

Please visit the [deer-flow-web](./web/) directory for more details.


## Supported Search Engines

Deer supports multiple search engines that can be configured in your `.env` file using the `SEARCH_API` variable:

- **Tavily** (default): A specialized search API for AI applications
    - Requires `TAVILY_API_KEY` in your `.env` file
    - Sign up at: https://app.tavily.com/home

- **DuckDuckGo**: Privacy-focused search engine
    - No API key required

- **Brave Search**: Privacy-focused search engine with advanced features
    - Requires `BRAVE_SEARCH_API_KEY` in your `.env` file
    - Sign up at: https://brave.com/search/api/

- **Arxiv**: Scientific paper search for academic research
    - No API key required
    - Specialized for scientific and academic papers

To configure your preferred search engine, set the `SEARCH_API` variable in your `.env` file:

```bash
# Choose one: tavily, duckduckgo, brave_search, arxiv
SEARCH_API=tavily
```

## Development

### Testing

Run the test suite:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/integration/test_workflow.py

# Run with coverage
make coverage
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format
```

## Architecture

DeerFlow implements a modular multi-agent system architecture designed for automated research and code analysis. The system is built on LangGraph, enabling a flexible state-based workflow where components communicate through a well-defined message passing system.

![Architecture Diagram](./assets/architecture.png)

The system employs a streamlined workflow with the following components:

1. **Coordinator**: The entry point that manages the workflow lifecycle
   - Initiates the research process based on user input
   - Delegates tasks to the planner when appropriate
   - Acts as the primary interface between the user and the system

2. **Planner**: Strategic component for task decomposition and planning
   - Analyzes research objectives and creates structured execution plans
   - Determines if enough context is available or if more research is needed
   - Manages the research flow and decides when to generate the final report

3. **Research Team**: A collection of specialized agents that execute the plan:
   - **Researcher**: Conducts web searches and information gathering using tools like Tavily and web crawling
   - **Coder**: Handles code analysis, execution, and technical tasks using Python REPL tool
   Each agent has access to specific tools optimized for their role and operates within the LangGraph framework

4. **Reporter**: Final stage processor for research outputs
   - Aggregates findings from the research team
   - Processes and structures the collected information
   - Generates comprehensive research reports

## Examples

The following examples demonstrate the capabilities of DeerFlow:

### Research Reports

1. **OpenAI Sora Report** - Analysis of OpenAI's Sora AI tool
   - Discusses features, access, prompt engineering, limitations, and ethical considerations
   - [View full report](examples/openai_sora_report.md)

2. **Google's Agent to Agent Protocol Report** - Overview of Google's Agent to Agent (A2A) protocol
   - Discusses its role in AI agent communication and its relationship with Anthropic's Model Context Protocol (MCP)
   - [View full report](examples/what_is_agent_to_agent_protocol.md)

3. **What is MCP?** - A comprehensive analysis of the term "MCP" across multiple contexts
   - Explores Model Context Protocol in AI, Monocalcium Phosphate in chemistry, and Micro-channel Plate in electronics
   - [View full report](examples/what_is_mcp.md)

4. **Bitcoin Price Fluctuations** - Analysis of recent Bitcoin price movements
   - Examines market trends, regulatory influences, and technical indicators
   - Provides recommendations based on historical data
   - [View full report](examples/bitcoin_price_fluctuation.md)

5. **What is LLM?** - An in-depth exploration of Large Language Models
   - Discusses architecture, training, applications, and ethical considerations
   - [View full report](examples/what_is_llm.md)

6. **How to Use Claude for Deep Research?** - Best practices and workflows for using Claude in deep research
   - Covers prompt engineering, data analysis, and integration with other tools
   - [View full report](examples/how_to_use_claude_deep_research.md)

7. **AI Adoption in Healthcare: Influencing Factors** - Analysis of factors driving AI adoption in healthcare
   - Discusses AI technologies, data quality, ethical considerations, economic evaluations, organizational readiness, and digital infrastructure
   - [View full report](examples/AI_adoption_in_healthcare.md)

8. **Quantum Computing Impact on Cryptography** - Analysis of quantum computing's impact on cryptography
   - Discusses vulnerabilities of classical cryptography, post-quantum cryptography, and quantum-resistant cryptographic solutions
   - [View full report](examples/Quantum_Computing_Impact_on_Cryptography.md)

9. **Cristiano Ronaldo's Performance Highlights** - Analysis of Cristiano Ronaldo's performance highlights
   - Discusses his career achievements, international goals, and performance in various matches
   - [View full report](examples/Cristiano_Ronaldo's_Performance_Highlights.md)

To run these examples or create your own research reports, you can use the following commands:

```bash
# Run with a specific query
uv run main.py "What factors are influencing AI adoption in healthcare?"

# Run with custom planning parameters
uv run main.py --max_plan_iterations 3 "How does quantum computing impact cryptography?"

# Run in interactive mode with built-in questions
uv run main.py --interactive

# Or run with basic interactive prompt
uv run main.py

# View all available options
uv run main.py --help
```

### Interactive Mode

The application now supports an interactive mode with built-in questions in both English and Chinese:

1. Launch the interactive mode:
   ```bash
   uv run main.py --interactive
   ```

2. Select your preferred language (English or 中文)

3. Choose from a list of built-in questions or select the option to ask your own question

4. The system will process your question and generate a comprehensive research report

### Human in the Loop

DeerFlow includes a human in the loop mechanism that allows you to review, edit, and approve research plans before they are executed:

1. **Plan Review**: When human in the loop is enabled, the system will present the generated research plan for your review before execution

2. **Providing Feedback**: You can:
   - Accept the plan by responding with `[ACCEPTED]`
   - Edit the plan by providing feedback (e.g., `[EDIT PLAN] Add more steps about technical implementation`)
   - The system will incorporate your feedback and generate a revised plan

3. **Auto-acceptance**: You can enable auto-acceptance to skip the review process:
   - Via API: Set `auto_accepted_plan: true` in your request

4. **API Integration**: When using the API, you can provide feedback through the `feedback` parameter:
   ```json
   {
     "messages": [{"role": "user", "content": "What is quantum computing?"}],
     "thread_id": "my_thread_id",
     "auto_accepted_plan": false,
     "feedback": "[EDIT PLAN] Include more about quantum algorithms"
   }
   ```

### Command Line Arguments

The application supports several command-line arguments to customize its behavior:

- **query**: The research query to process (can be multiple words)
- **--interactive**: Run in interactive mode with built-in questions
- **--max_plan_iterations**: Maximum number of planning cycles (default: 1)
- **--max_step_num**: Maximum number of steps in a research plan (default: 3)
- **--debug**: Enable detailed debug logging

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to all the open source projects and contributors that make DeerFlow possible. We stand on the shoulders of giants.

In particular, we want to express our deep appreciation for:
- [LangChain](https://github.com/langchain-ai/langchain) for their exceptional framework that powers our LLM interactions and chains
- [LangGraph](https://github.com/langchain-ai/langgraph) for enabling our sophisticated multi-agent orchestration

These amazing projects form the foundation of DeerFlow and demonstrate the power of open source collaboration.
