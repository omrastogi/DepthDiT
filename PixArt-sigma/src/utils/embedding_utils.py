import os
import torch
from transformers import T5Tokenizer, T5EncoderModel

def save_null_caption_embeddings(pipeline_path, accelerator, save_dir="output/null_embedding"):
    """Save the null caption token and its embeddings to .pt files."""
    os.makedirs(save_dir, exist_ok=True)

    max_sequence_length = 300
    tokenizer = T5Tokenizer.from_pretrained(pipeline_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        pipeline_path, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(accelerator.device)

    null_caption_token = tokenizer(
        "", max_length=max_sequence_length, padding="max_length", 
        truncation=True, return_tensors="pt"
    ).to(accelerator.device)

    null_caption_embs = text_encoder(
        null_caption_token.input_ids, 
        attention_mask=null_caption_token.attention_mask
    )[0]

    torch.save(null_caption_token, os.path.join(save_dir, "null_caption_token.pt"))
    torch.save(null_caption_embs, os.path.join(save_dir, "null_caption_embs.pt"))

    del tokenizer, text_encoder
    torch.cuda.empty_cache()

    print(f"Saved null caption token and embeddings to {save_dir}")


def load_null_caption_embeddings(save_dir="output/null_embedding"):
    """Load the saved null caption token and its embeddings from .pt files."""
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The directory {save_dir} does not exist.")

    null_caption_token = torch.load(os.path.join(save_dir, "null_caption_token.pt"))
    null_caption_embs = torch.load(os.path.join(save_dir, "null_caption_embs.pt"))

    print(f"Loaded null caption token and embeddings from {save_dir}")
    return null_caption_token, null_caption_embs


