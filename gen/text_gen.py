from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os

from .utils import GenerationConfig, device_str

DEFAULT_HF_MODEL = "HuggingFaceH4/zephyr-7b-alpha"
OPENAI_MODEL = "gpt-4o-mini"

@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str

class TextGenerator:
    def __init__(self, use_openai: bool = False, hf_model: str = DEFAULT_HF_MODEL):
        self.use_openai = use_openai
        self.hf_model = hf_model
        self._init_backend()

    def _init_backend(self):
        if self.use_openai:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.backend = "openai"
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model,
                torch_dtype=getattr(torch, "float16", None) or getattr(torch, "bfloat16", None) or torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.backend = f"hf:{self.hf_model} ({device_str()})"

    def generate(self, messages: List[ChatMessage], config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        if self.use_openai:
            # OpenAI chat.completions
            payload = [ {"role": m.role, "content": m.content} for m in messages ]
            res = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=payload,
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
            )
            return res.choices[0].message.content

        # HF local generation using a chat template if available
        from transformers import pipeline
        from transformers import AutoTokenizer
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False, add_generation_prompt=True
            )
        else:
            # naive fallback
            prompt = ""
            for m in messages:
                prompt += f"[{m.role.upper()}]\n{m.content}\n\n"
            prompt += "[ASSISTANT]\n"

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        out = pipe(
            prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=True if config.temperature > 0 else False,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()
