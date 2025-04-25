# 🦌 DeerFlow

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Originated from Open Source, give back to Open Source.

**DeerFlow** (**D**eep **E**xploration and **E**fficient **R**esearch **Flow**) is a community-driven Deep Research framework that builds upon the incredible work of the open source community. Our goal is to combine language models with specialized tools for tasks like web search, crawling, and Python code execution, while giving back to the community that made this possible.

## 🎬 Demo
> [!NOTE]
> The demo video is not uploaded yet.

## 📑 Table of Contents
- [🚀 Quick Start](#quick-start)
- [🛠️ Development](#development)
- [🏗️ Architecture](#architecture)
- [🗣️ Text-to-Speech Integration](#text-to-speech-integration)
- [📚 Examples](#examples)
- [💖 Acknowledgments](#acknowledgments)


## Quick Start

DeerFlow is developed in Python, and comes with a web UI written in Node.js. To ensure a smooth setup process, we recommend using the following tools:

### Recommended Tools
- **[`uv`](https://docs.astral.sh/uv/getting-started/installation/):**
  Simplify Python environment and dependency management. `uv` automatically creates a virtual environment in the root directory and installs all required packages for you—no need to manually install Python environments.

- **[`nvm`](https://github.com/nvm-sh/nvm):**
  Manage multiple versions of the Node.js runtime effortlessly.

- **[`pnpm`](https://pnpm.io/installation):**
  Install and manage dependencies of Node.js project.

### Environment Requirements
Make sure your system meets the following minimum requirements:
- **[Python](https://www.python.org/downloads/):** Version `3.12+`
- **[Node.js](https://nodejs.org/en/download/):** Version `22+`

### Installation
```bash
# Clone the repository
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow

# Install dependencies, uv will take care of the python interpreter and venv creation, and install the required packages
uv sync

# Configure .env with your API keys
# Tavily: https://app.tavily.com/home
# Brave_SEARCH: https://brave.com/search/api/
# volcengine TTS: Add your TTS credentials if you have them
cp .env.example .env

# See the 'Supported Search Engines' and 'Text-to-Speech Integration' sections below for all available options

# Configure conf.yaml for your LLM model and API keys
# Gemini: https://ai.google.dev/gemini-api/docs/openai
cp conf.yaml.example conf.yaml

# Install marp for ppt generation
# https://github.com/marp-team/marp-cli?tab=readme-ov-file#use-package-manager
brew install marp-cli
```

Optionally, install web UI dependencies via [pnpm](https://pnpm.io/installation):

```bash
cd web
pnpm install
```

### Console UI

The quickest way to run the project is to use the console UI.

```bash
# Run the project in a bash-like shell
uv run main.py
```

### Web UI

This project also includes a Web UI, offering a more dynamic and engaging interactive experience.
> Note: You need to install the dependencies of web UI first.

```bash
# Run both the backend and frontend servers
./bootstrap.sh
```

Open your browser and visit [`http://localhost:3000`](http://localhost:3000) to explore the web UI.

Please visit the [`web`](./web/) directory for more details.


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

### Debugging with LangGraph Studio

DeerFlow uses LangGraph for its workflow architecture. You can use LangGraph Studio to debug and visualize the workflow in real-time.

#### Running LangGraph Studio Locally

DeerFlow includes a `langgraph.json` configuration file that defines the graph structure and dependencies for the LangGraph Studio. This file points to the workflow graphs defined in the project and automatically loads environment variables from the `.env` file.

##### Mac

```bash
# Install uv package manager if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking
```

##### Windows / Linux

```bash
# Install dependencies
pip install -e .
pip install -U "langgraph-cli[inmem]"

# Start the LangGraph server
langgraph dev
```

After starting the LangGraph server, you'll see several URLs in the terminal:
- API: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API Docs: http://127.0.0.1:2024/docs

Open the Studio UI link in your browser to access the debugging interface.

#### Using LangGraph Studio

In the Studio UI, you can:

1. Visualize the workflow graph and see how components connect
2. Trace execution in real-time to see how data flows through the system
3. Inspect the state at each step of the workflow
4. Debug issues by examining inputs and outputs of each component
5. Provide feedback during the planning phase to refine research plans

When you submit a research topic in the Studio UI, you'll be able to see the entire workflow execution, including:
- The planning phase where the research plan is created
- The feedback loop where you can modify the plan
- The research and writing phases for each section
- The final report generation

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

## Text-to-Speech Integration

DeerFlow now includes a Text-to-Speech (TTS) feature that allows you to convert research reports to speech. This feature uses the volcengine TTS API to generate high-quality audio from text.

### Features

- Convert any text or research report to natural-sounding speech
- Adjust speech parameters like speed, volume, and pitch
- Support for multiple voice types
- Available through both API and web interface

### Using the TTS API

You can access the TTS functionality through the `/api/tts` endpoint:

```bash
# Example API call using curl
curl --location 'http://localhost:8000/api/tts' \
--header 'Content-Type: application/json' \
--data '{
    "text": "This is a test of the text-to-speech functionality.",
    "speed_ratio": 1.0,
    "volume_ratio": 1.0,
    "pitch_ratio": 1.0
}' \
--output speech.mp3
```

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

DeerFlow is built upon the incredible work of the open-source community. We are deeply grateful to all the projects and contributors whose efforts have made DeerFlow possible. Truly, we stand on the shoulders of giants.

We would like to extend our sincere appreciation to the following projects for their invaluable contributions:

- **[LangChain](https://github.com/langchain-ai/langchain)**: Their exceptional framework powers our LLM interactions and chains, enabling seamless integration and functionality.
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Their innovative approach to multi-agent orchestration has been instrumental in enabling DeerFlow's sophisticated workflows.

These projects exemplify the transformative power of open-source collaboration, and we are proud to build upon their foundations.

### Key Contributors
A heartfelt thank you goes out to the core authors of `DeerFlow`, whose vision, passion, and dedication have brought this project to life:

- **[Daniel Walnut](https://github.com/hetaoBackend/)**
- **[Henry Li](https://github.com/magiccube/)**

Your unwavering commitment and expertise have been the driving force behind DeerFlow's success. We are honored to have you at the helm of this journey.
