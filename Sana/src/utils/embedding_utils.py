import os
from os import path as osp
import torch
from transformers import T5Tokenizer, T5EncoderModel

import os
import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForCausalLM

def save_null_caption_embeddings(encoder_name, max_sequence_length, device, save_dir="output/null_embedding"):
    """Save the null caption token and its embeddings to .pt files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Dynamically initialize tokenizer and text encoder
    text_encoder_dict = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "google/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }
    
    assert encoder_name in text_encoder_dict, f"Unsupported encoder type: {encoder_name}"
    
    if "T5" in encoder_name:
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_dict[encoder_name])
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_dict[encoder_name], torch_dtype=torch.float16).to(device)
    elif "gemma" in encoder_name or "Qwen" in encoder_name:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dict[encoder_name])
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(text_encoder_dict[encoder_name], torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_name}")
    
    # Generate null caption tokens and embeddings
    null_caption_token = tokenizer(
        "", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    
    if "T5" in encoder_name:
        null_caption_embs = text_encoder(
            null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask
        )[0]
    elif "gemma" in encoder_name or "Qwen" in encoder_name:
        null_caption_embs = text_encoder(
            null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask
        )[0]
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_name}")
    
    # Save embeddings and tokens to files
    torch.save(null_caption_token, os.path.join(save_dir, "null_caption_token.pt"))
    torch.save(null_caption_embs, os.path.join(save_dir, "null_caption_embs.pt"))
    torch.save(
        {'uncond_prompt_embeds': null_caption_embs, 'uncond_prompt_embeds_mask': null_caption_token.attention_mask},
        os.path.join(save_dir, f"null_embed_{encoder_name}_{max_sequence_length}token.pth")
    )
    
    # Clean up
    del null_caption_embs, null_caption_token, tokenizer, text_encoder
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


