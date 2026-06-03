def get_think_tags(model_name):
    """Return the appropriate think tokens based on the model name."""
    if "gemma-4" in model_name:
        return {"open": "<|channel>thought", "close": "<channel|>"}
    else:
        return {"open": "<think>", "close": "</think>"}

def init_llm_server(model_name, *args, **kwargs):
    """Initialize LLM server configuration based on the model name."""
    if "gemma-4" in model_name:
        return _init_llm_server_gemma4(model_name, *args, **kwargs)
    else:
        return _init_llm_server_default(model_name, *args, **kwargs)

def _init_llm_server_gemma4(model_name, context_length=32768, port=8000):
    """Initialize LLM server configuration."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "context_length": context_length,
        "stream": True,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42,
        "skip_special_tokens": False,
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}

def _init_llm_server_default(model_name, context_length=32768, port=8000):
    """Initialize LLM server configuration."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "context_length": context_length,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}
