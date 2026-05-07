import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_model(model_dir="models", output_dir="models_int8"):
    """Apply dynamic INT8 quantization to a saved model."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    original_size = sum(p.numel() * p.element_size() for p in model.parameters())

    print("  Applying INT8 dynamic quantization...")
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    tokenizer.save_pretrained(output_dir)
    quantized.save_pretrained = None  # quantized models need torch.save
    torch.save(quantized, os.path.join(output_dir, "model_int8.pt"))
    tokenizer.save_pretrained(output_dir)

    quantized_size = os.path.getsize(os.path.join(output_dir, "model_int8.pt"))

    print(f"  Original:  {original_size / (1024*1024):.2f} MB")
    print(f"  Quantized: {quantized_size / (1024*1024):.2f} MB")
    print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"  Saved to {output_dir}/")


if __name__ == "__main__":
    quantize_model()