# AQGPT

AQGPT uses:

- local pollution/source/weather functions
- an UrbanEmissions RAG
- Qwen served through Ollama for query routing, summarization, and RAG answer generation

## Current local model setup

The repo is configured for a local Ollama server:

```env
OLLAMA_BASE_URL=http://localhost:11434
AQGPT_TEXT_PROVIDER=qwen
AQGPT_TEXT_MODEL=qwen2.5:72b
AQGPT_FUNCTION_PROVIDER=qwen
AQGPT_FUNCTION_MODEL=qwen2.5:72b
AQGPT_RAG_PROVIDER=qwen
AQGPT_RAG_MODEL=qwen2.5:72b
```

Start Ollama if it is not already running:

```bash
bash ./scripts/start_ollama_local.sh
```

If your main server already has Ollama and `qwen2.5:72b`, you only need the Ollama process running and reachable at `OLLAMA_BASE_URL`.
