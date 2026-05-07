import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name, output_dir="models"):
    """Download a HuggingFace model for local inference."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    print(f"  Model saved to {output_dir}/ ({size_mb:.1f} MB)")
    return output_dir


if __name__ == "__main__":
    download_model("sshleifer/tiny-gpt2")