"""
File taken verbatim from the original repo (https://github.com/thu-coai/Agent-SafetyBench), file 
Agent-SafetyBench/evaluation/model_api/__init__.py, with the following changes:
1. commented out some imports due to errors
2. Added import for VLLM support
"""

from .GLM4API import GLM4API
# from .InternlmAPI import InternlmAPI  
from .Llama3API import Llama3API
# from .MistralAPI import MistralAPI  
from .OpenaiAPI import OpenaiAPI
from .QwenAPI import QwenAPI
from .ClaudeAPI import ClaudeAPI
from .GeminiAPI import GeminiAPI
from .DeepseekAPI import DeepseekAPI
from .QwenCloudAPI import QwenCloudAPI
# from .MistralCloudAPI import MistralCloudAPI  
from .LlamaCloudAPI import LlamaCloudAPI
from .VllmAPI import VllmAPI
# from .VllmLlamaAPI import VllmLlamaAPI
