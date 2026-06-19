import asyncio
import logging
from transformers import AutoTokenizer
from interwhen.monitors import SimpleTextReplaceMonitor
from interwhen import stream_completion
from interwhen.utils.llm import init_llm_server, get_think_tags

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )
    model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    think_tags = get_think_tags(model_name)
    llm_server = init_llm_server(model_name, context_length=200)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prepare the model input
    prompt = "Explain quantum computing in simple terms."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    result = asyncio.run(stream_completion(
        text,
        llm_server=llm_server,
        monitors=(SimpleTextReplaceMonitor("IsCheck", think_tags['close'], async_execution=True),),
        add_delay=False,
        termination_requires_validation=False,
        async_execution=True,
        tokenizer=tokenizer
    ))
    
    # Save output to file
    output_file = "../output.txt"
    with open(output_file, "w") as f:
        f.write(result)
    print(f"\n\nOutput saved to {output_file}")
