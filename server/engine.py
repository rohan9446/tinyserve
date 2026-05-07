import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import sample


class InferenceEngine:
    """Loads a model and generates tokens using the C++ sampler."""

    def __init__(self, model_dir="models"):
        print(f"  Loading model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {param_count:,} parameters")

    def generate(self, prompt, max_tokens=50, temperature=0.7, top_k=50, top_p=0.9):
        """Generate text token-by-token, yielding each token as it's produced."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated = input_ids[0].tolist()

        ttft = None
        start = time.perf_counter()
        seed = int(time.perf_counter() * 1000) % (2**31)

        with torch.no_grad():
            for i in range(max_tokens):
                inputs = torch.tensor([generated])
                outputs = self.model(inputs)
                logits = outputs.logits[0, -1, :].numpy()

                token_id = sample(logits, temperature, top_k, top_p, seed + i)

                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000

                generated.append(token_id)
                token_text = self.tokenizer.decode([token_id])

                if token_id == self.tokenizer.eos_token_id:
                    break

                yield token_text

        total_time = (time.perf_counter() - start) * 1000
        tokens_generated = len(generated) - len(input_ids[0])
        tokens_per_sec = tokens_generated / (total_time / 1000) if total_time > 0 else 0

        yield f"\n\n[TTFT: {ttft:.1f}ms | Tokens: {tokens_generated} | Speed: {tokens_per_sec:.1f} tok/s]"