# Prompt Optimizer Agent

AI agent that optimizes prompts and tests them across multiple LLMs using MCP (Model Context Protocol) servers.

## Architecture

```
prompt-optimizer-agent/
├── mcp-servers/                  # MCP server implementations
│   ├── llm-router/server.py      # Routes prompts to Claude, GPT, Gemini, Perplexity
│   ├── prompt-analyzer/server.py # Analyzes and optimizes prompt quality
│   └── results-evaluator/server.py # Compares and scores LLM outputs
├── agent-core/                   # Core agent logic
│   ├── main.py                   # CLI entry point and orchestrator
│   ├── optimizer.py              # Optimization pipeline
│   └── config.py                 # Configuration loader
├── config/
│   └── agent_config.yaml         # Default settings
├── .env.example                  # API key template
└── requirements.txt              # Python dependencies
```

### How It Works

1. **Prompt Analyzer** scores your prompt on clarity, specificity, structure, and completeness, then generates an improved version with concrete suggestions.
2. **LLM Router** sends the optimized prompt to multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini, Perplexity) in parallel.
3. **Results Evaluator** compares the responses on relevance, coherence, depth, and accuracy signals, then ranks the models.

The optimizer runs multiple rounds of this pipeline until the prompt quality score reaches 90+ or the configured number of rounds is exhausted.

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd prompt-optimizer-agent
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
PERPLEXITY_API_KEY=pplx-...
```

You only need keys for the providers you want to use. The agent gracefully handles missing keys.

### 3. (Optional) Edit configuration

Adjust `config/agent_config.yaml` to change default model parameters, cost rates, optimization rounds, or logging settings.

## Usage

### CLI Commands

```bash
# Analyze prompt quality (scores 0-100 on four dimensions)
python -m agent_core.main analyze "Write me a story"

# Optimize a prompt (iterative improvement)
python -m agent_core.main optimize "Write me a story"

# Optimize with a specific task type and 5 rounds
python -m agent_core.main optimize --task-type creative_writing --rounds 5 "Write me a story"

# Optimize AND test across all LLMs
python -m agent_core.main optimize --test "Explain quantum computing step by step"

# Test a prompt across all LLM providers
python -m agent_core.main test "What are the benefits of Rust over C++?"

# Get recommended parameters for a task type
python -m agent_core.main params code_generation

# Get a system prompt suggestion
python -m agent_core.main system-prompt data_analysis

# Verbose output (full JSON)
python -m agent_core.main -v optimize "Summarize this article"
```

### Supported Task Types

| Task Type | Temperature | Max Tokens | Best For |
|---|---|---|---|
| `creative_writing` | 0.9 | 2048 | Fiction, poetry, storytelling |
| `code_generation` | 0.2 | 4096 | Writing or debugging code |
| `data_analysis` | 0.3 | 2048 | Datasets, statistics, insights |
| `summarization` | 0.3 | 1024 | Condensing long texts |
| `question_answering` | 0.4 | 1024 | Factual Q&A, research |
| `translation` | 0.3 | 2048 | Language translation |
| `general` | 0.7 | 1024 | General-purpose assistance |

### Running MCP Servers Standalone

Each MCP server can be run independently:

```bash
python mcp-servers/llm-router/server.py
python mcp-servers/prompt-analyzer/server.py
python mcp-servers/results-evaluator/server.py
```

### Using as a Library

```python
import asyncio
from agent_core.optimizer import PromptOptimizer

async def main():
    optimizer = PromptOptimizer()

    # Analyze a prompt
    analysis = await optimizer.analyze("Tell me about dogs")
    print(f"Quality score: {analysis['overall_score']}/100")

    # Optimize a prompt
    result = await optimizer.optimize(
        prompt="Tell me about dogs",
        task_type="question_answering",
        rounds=3,
    )
    print(f"Improved prompt: {result['final_prompt']}")

    # Test across all LLMs
    test_result = await optimizer.test_prompt("Explain quantum entanglement")
    for entry in test_result.get("evaluation", {}).get("ranking", []):
        print(f"#{entry['rank']} {entry['provider']} — score {entry['score']}")

asyncio.run(main())
```

## MCP Server Tools

### llm-router

| Tool | Description |
|---|---|
| `call_claude()` | Send prompt to Anthropic Claude |
| `call_gpt()` | Send prompt to OpenAI GPT |
| `call_gemini()` | Send prompt to Google Gemini |
| `call_perplexity()` | Send prompt to Perplexity |
| `call_all_llms()` | Send prompt to all providers in parallel |

### prompt-analyzer

| Tool | Description |
|---|---|
| `analyze_prompt_quality()` | Score clarity, specificity, structure, completeness (0-100) |
| `optimize_prompt()` | Generate improvement suggestions and a rewritten prompt |
| `suggest_system_prompt()` | Generate a system prompt for a task type |
| `suggest_model_parameters()` | Recommend temperature and max_tokens for a task type |

### results-evaluator

| Tool | Description |
|---|---|
| `compare_responses()` | Side-by-side comparison of multiple LLM responses |
| `score_response_quality()` | Score a single response on quality metrics |
| `recommend_best_model()` | Pick the optimal model for a task type and priority set |

## Configuration

`config/agent_config.yaml` controls:

- **model_defaults** — default temperature, max_tokens, provider, and model
- **cost** — per-provider token pricing for cost estimation
- **logging** — log level, format, optional log file
- **mcp_servers** — registered MCP server definitions
- **optimization_rounds** — how many improve-evaluate cycles to run (default 3)
- **parallel_evaluation** — whether to call all providers concurrently

## License

MIT
