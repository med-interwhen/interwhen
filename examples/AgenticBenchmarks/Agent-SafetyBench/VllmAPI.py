"""
Content of this file is taken from the original code files in the Agent-SafetyBench repo (https://github.com/thu-coai/Agent-SafetyBench), at
Agent-SafetyBench/evaluation/model_api/QwenAPI.py and Agent-SafetyBench/evaluation/model_api/OpenaiAPI.py.
We made some changes to accommodate VLLM usage.

Most of the VllmAPI.generate_response code is verbatim from QwenAPI. We made some changes to incorporate reasoning models.
Most of the VllmAPI.response code is verbatim from OpenaiAPI, with some changes.

"""from openai import OpenAI
from transformers import AutoTokenizer
import time
import json
import random
import re
import string
import sys
sys.path.append('./model_api')
from BaseAPI import BaseAPI

class VllmAPI(BaseAPI):
    def __init__(self, model_name, base_url="http://localhost:8000/v1", tokenizer_name=None, generation_config={}):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
        self.sys_prompt = self.sys_prompt_with_failure_modes
        # self.sys_prompt = self.basic_sys_prompt

    def response(self, messages, tools, skip_thinking=False, max_tokens=None):
        messages = [   
        {**m, "content": (m.get("content") or "")}
        for m in messages
        ]
        tpl_kwargs = dict(add_generation_prompt=True, tokenize=False)
        if tools:
            tpl_kwargs["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **tpl_kwargs)
        if skip_thinking:
            prompt += "<think>\n\n</think>\n\n"
        # Per-call config override; do NOT mutate self.generation_config (it is
        # shared across worker threads).
        gen_cfg = self.generation_config if max_tokens is None else {
            **self.generation_config, "max_tokens": max_tokens
        }
        for _ in range(10):
            try:
                completion = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    **gen_cfg
                )
                if completion is None or completion.choices is None:
                    continue
                tokens = completion.usage.completion_tokens if completion.usage else 0
                return completion.choices[0].text, tokens
            except Exception as e:
                # print(e)
                time.sleep(1)
        return None, 0

    def generate_response(self, messages, tools, skip_thinking=False, max_tokens=None):
        completion, tokens = self.response(messages, tools, skip_thinking=skip_thinking, max_tokens=max_tokens)

        if completion is None: return None

        # Strip thinking blocks (QwQ, etc.)
        completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL).strip()
        if "</think>" in completion:
            completion = completion[completion.rfind("</think>") + len("</think>"):].strip()
        else:
            completion = completion.strip()

        ## tool call part — parse <tool_call> tags, same as QwenAPI
        if '<tool_call>' in completion:
            completion = completion[:completion.find('</tool_call>')].replace('<tool_call>', '').strip()
            try:
                start = completion.index('{')
                end = completion.rindex('}')
            except ValueError:
                return {'type': 'error', 'message': f'No JSON found in tool call: {completion}'}
            completion = completion[start:end + 1]
            completion = completion.replace('\n', '\\n')
            if self.is_json(completion):
                res = self.parse_json(completion)
                if 'name' not in res or 'arguments' not in res:
                    return {'type': 'error', 'message': f'Wrong tool call result: {res}'}
                tool_call_id = ''.join(random.sample(string.ascii_letters + string.digits, 9))
                tool_name = res['name']
                if isinstance(res['arguments'], dict):
                    arguments = res['arguments']
                    return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments, 'tokens': tokens}
                elif self.is_json(res['arguments']):
                    arguments = self.parse_json(res['arguments'])
                    return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments, 'tokens': tokens}
                else:
                    return {'type': 'error', 'message': f"Wrong argument format: {res['arguments']}", 'tokens': tokens}
            else:
                return {'type': 'error', 'message': f'Wrong tool call result: {completion}', 'tokens': tokens}

        ## normal content part
        else:
            return {'type': 'content', 'content': completion, 'tokens': tokens}
