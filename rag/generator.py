import importlib
import logging
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_path: Optional[str] = None,
        provider: str = "hf",
        dtype: torch.dtype = torch.float16,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        trust_remote_code: bool = True,
        device: Optional[Union[str, int]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_path = tokenizer_path or model_name_or_path
        self.provider = provider
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.generator = None
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_vllm = provider == "vllm" and importlib.util.find_spec("vllm") is not None

        if self.use_vllm:
            from vllm import LLM, SamplingParams

            if isinstance(dtype, torch.dtype):
                vllm_dtype = getattr(dtype, "name", str(dtype).replace("torch.", ""))
            else:
                vllm_dtype = dtype
            self.sampling_params = SamplingParams()
            self.llm = LLM(
                model=model_name_or_path,
                tokenizer=self.tokenizer_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                dtype=vllm_dtype,
                max_model_len=max_model_len,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            device_map = device if device is not None else "auto"
            quant_config = None
            if load_in_8bit and load_in_4bit:
                raise ValueError("Only one of load_in_8bit or load_in_4bit can be True")
            if load_in_8bit:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                dtype = None
            elif load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                dtype = None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                quantization_config=quant_config,
            )
            self.generator = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device_map,
            )

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Union[str, List[str]]:
        prompts: List[str] = [prompt] if isinstance(prompt, str) else prompt

        if self.use_vllm:
            from vllm import SamplingParams

            params = SamplingParams(
                temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=[]
            )
            outputs = self.llm.generate(prompts, sampling_params=params)
            texts = [out.outputs[0].text.strip() for out in outputs]
        else:
            responses = self.generator(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            texts = []
            for prompt_text, resp in zip(prompts, responses):
                # pipeline may return a list of dicts per item
                if isinstance(resp, list):
                    resp = resp[0]
                generated = resp["generated_text"]
                if isinstance(generated, list):
                    generated = generated[0]
                cleaned = generated[len(prompt_text) :].strip() if generated.startswith(prompt_text) else generated.strip()
                texts.append(cleaned)

        if isinstance(prompt, str):
            return texts[0]
        return texts
