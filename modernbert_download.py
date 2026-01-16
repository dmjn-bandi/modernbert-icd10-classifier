from transformers import AutoTokenizer, AutoModel
import os

model_name = "answerdotai/ModernBERT-base"

local_save_path = "./models/base/ModernBERT-base"

try:
    print(f"Starting download for: {model_name} ...")

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Downloading model weights...")
    model = AutoModel.from_pretrained(model_name)

    os.makedirs(local_save_path, exist_ok=True)

    print("Saving model and tokenizer...")
    tokenizer.save_pretrained(local_save_path)
    model.save_pretrained(local_save_path)

    print("\nSuccess! Download and save completed.")
    print(f"Full path: {os.path.abspath(local_save_path)}")

except Exception as e:
    print(e)
