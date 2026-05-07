import time
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import sample


class InferenceEngine:
    """Loads a model and generates tokens using the C++ sampler."""

    def __init__(self, model_dir="models"):
        print(f"  Loading model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        qt_path = os.path.join(model_dir, "model_int8.pt")
        if os.path.exists(qt_path):
            print("  Loading quantized INT8 model...")
            self.model = torch.load(qt_path, map_location="cpu", weights_only=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir)

        self.model.eval()
        self.model_dir = model_dir

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        param_count = sum(p.numel() for p in self.model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        print(f"  Model loaded: {param_count:,} parameters ({size_mb:.1f} MB)")

    def _format_prompt(self, prompt):
        """Apply chat template if the model supports it."""
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(self, prompt, max_tokens=50, temperature=0.7, top_k=50, top_p=0.9):
        """Generate text token-by-token, yielding each token and returning stats."""
        formatted = self._format_prompt(prompt)
        input_ids = self.tokenizer.encode(formatted, return_tensors="pt")
        generated = input_ids[0].tolist()
        prompt_len = len(generated)

        ttft = None
        start = time.perf_counter()
        seed = int(time.perf_counter() * 1000) % (2**31)
        prev_text = ""

        with torch.no_grad():
            for i in range(max_tokens):
                inputs = torch.tensor([generated])
                outputs = self.model(inputs)
                logits = outputs.logits[0, -1, :].numpy()

                token_id = sample(logits, temperature, top_k, top_p, seed + i)

                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000

                generated.append(token_id)

                if token_id == self.tokenizer.eos_token_id:
                    break

                # decode all generated tokens to get proper spacing
                full_text = self.tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
                new_chars = full_text[len(prev_text):]
                prev_text = full_text

                if new_chars:
                    yield {"type": "token", "text": new_chars}

        total_time = (time.perf_counter() - start) * 1000
        tokens_generated = len(generated) - prompt_len
        tokens_per_sec = tokens_generated / (total_time / 1000) if total_time > 0 else 0

        yield {
            "type": "stats",
            "ttft_ms": round(ttft, 1) if ttft else 0,
            "tokens": tokens_generated,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "total_ms": round(total_time, 1),
        }