def get_think_tags(model_name):
    """Return the appropriate think tokens based on the model name."""
    if "gemma-4" in model_name:
        return {"open": "<|channel>thought", "close": "<channel|>"}
    else:
        return {"open": "<think>", "close": "</think>"}


def render_user_turn(tokenizer, content, assistant_seed=""):
    """Render a mid-stream user turn to append onto an ongoing assistant generation.

    Returns the text that closes the current (in-progress) assistant turn, adds a
    user message with ``content``, and reopens an assistant turn seeded with
    ``assistant_seed`` (e.g. the thinking-open tag).

    It does this by simply rendering the conversation with ``apply_chat_template``
    and taking the suffix that follows the running assistant text — so no chat
    tokens are hardcoded and any default system/BOS prefix (which only appears at
    the very start) is naturally excluded. A tokenizer is required.
    """
    if tokenizer is None:
        raise ValueError("render_user_turn requires a tokenizer")

    U = "\u0000USR\u0000"
    A = "\u0000AST\u0000"
    # Render through the assistant content...
    two = tokenizer.apply_chat_template(
        [{"role": "user", "content": U}, {"role": "assistant", "content": A}],
        tokenize=False,
        add_generation_prompt=False,
    )
    cut = two.index(A) + len(A)
    # ...then the same conversation plus the feedback as a real user turn.
    three = tokenizer.apply_chat_template(
        [{"role": "user", "content": U},
         {"role": "assistant", "content": A},
         {"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Suffix after the assistant content = assistant_end + user turn + assistant reopen.
    return three[cut:] + assistant_seed


def get_eot_token(tokenizer):
    """Return the model's end-of-(assistant)-turn marker (e.g. ChatML ``<|im_end|>``).

    Derived from the chat template so it works across model families. A tokenizer
    is required.
    """
    if tokenizer is None:
        raise ValueError("get_eot_token requires a tokenizer")
    A = "\u0000AST\u0000"
    two = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": A}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return two[two.index(A) + len(A):].strip()

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
