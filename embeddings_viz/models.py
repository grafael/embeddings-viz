"""Model loading, vocabulary extraction, and progress tracking."""

import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from embeddings_viz.config import (
    MODEL_TYPE_MAP,
    VOCAB_CACHE_DIR,
    VOCAB_FILE,
)

model_state = {}

# Progress tracker for async model loading (polled by the frontend).
progress = {
    "active": False,
    "step": 0,
    "total_steps": 3,
    "step_name": "",
    "detail": "",
    "percent": 0,
    "done": False,
    "error": None,
    "result": None,
}


def _update_progress(step, step_name, detail="", percent=0):
    progress["step"] = step
    progress["step_name"] = step_name
    progress["detail"] = detail
    progress["percent"] = percent



def _extract_vocab_from_tokenizer(tokenizer):
    """Extract clean whole words from a tokenizer's vocabulary."""
    vocab = tokenizer.get_vocab()
    total = len(vocab)
    words = set()
    for i, token in enumerate(tqdm(vocab, desc="Extracting vocabulary", unit="tok")):
        # Strip common subword prefixes (WordPiece, BPE, SentencePiece)
        clean = token.replace("##", "").replace("Ġ", "").replace("▁", "")
        if clean.isalpha() and len(clean) > 1:
            words.add(clean.lower())
        if i % 1000 == 0:
            _update_progress(2, "Extracting vocabulary", f"{i}/{total} tokens", int(i / total * 100))
    _update_progress(2, "Extracting vocabulary", f"{len(words)} words found", 100)
    return sorted(words)


def _load_vocab(tokenizer):
    """Load vocab from vocab.txt if present, otherwise extract from tokenizer."""
    if VOCAB_FILE.exists():
        with open(VOCAB_FILE) as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(words)} words from vocab.txt")
        return words
    words = _extract_vocab_from_tokenizer(tokenizer)
    print(f"Vocabulary: {len(words)} words")
    return words


def _vocab_cache_path(model_name):
    safe = model_name.replace("/", "_")
    return VOCAB_CACHE_DIR / f"{safe}_v3.npy"


def _format_eta(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"



def _encode_vocab(batch_size=64):
    """Encode all vocab words into the same representation space used for
    contextual word embeddings.

    Embedding models: run each word through the transformer and average the
    word's sub-tokens from the last hidden state (excluding CLS/SEP). This
    matches the extraction in get_contextual_word_embeddings.

    Generative models: use the input embedding matrix directly, since causal
    hidden states for isolated words are degenerate (all similarities ~1.0).
    """
    tokenizer = model_state["tokenizer"]
    transformer = model_state["transformer"]
    vocab_words = model_state["vocab_words"]
    total = len(vocab_words)
    t0 = time.monotonic()

    if model_state["type"] == "generative":
        embed_matrix = transformer.get_input_embeddings().weight.detach()
        all_embs = []
        for i in range(0, total, batch_size):
            for word in vocab_words[i : i + batch_size]:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                all_embs.append(embed_matrix[token_ids].mean(dim=0).cpu().numpy())
            done = min(i + batch_size, total)
            elapsed = time.monotonic() - t0
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            _update_progress(
                3, "Encoding vocabulary",
                f"{done}/{total} words — ETA {_format_eta(eta)}", int(done / total * 100),
            )
        return np.vstack(all_embs)

    # Embedding models: forward pass through transformer, average word tokens only
    device = transformer.device
    all_embs = []
    for i in range(0, total, batch_size):
        batch = vocab_words[i : i + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=64, return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = transformer(**encoded).last_hidden_state.cpu()
        for j in range(len(batch)):
            # Only average sub-tokens belonging to the word (exclude CLS/SEP)
            word_tokens = [t for t, (s, e) in enumerate(offsets[j]) if s != e]
            if word_tokens:
                all_embs.append(hidden[j, word_tokens].mean(dim=0).numpy())
            else:
                mask = encoded["attention_mask"][j].cpu().bool()
                all_embs.append(hidden[j, mask].mean(dim=0).numpy())
        done = min(i + batch_size, total)
        elapsed = time.monotonic() - t0
        eta = (elapsed / done) * (total - done) if done > 0 else 0
        _update_progress(
            3, "Encoding vocabulary",
            f"{done}/{total} words — ETA {_format_eta(eta)}", int(done / total * 100),
        )
    return np.vstack(all_embs)



def load_model(model_name):
    """Download (if needed) and initialize a model with its vocabulary embeddings."""
    model_type = MODEL_TYPE_MAP.get(model_name, "embedding")
    print(f"\n{'=' * 50}")
    print(f"Loading {model_name} ({model_type})")
    print(f"{'=' * 50}")

    model_state["name"] = model_name
    model_state["type"] = model_type

    # Step 1 — Download / load the model weights
    _update_progress(1, "Downloading model", model_name)
    print("\n[1/3] Downloading model...")
    if model_type == "generative":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        model_state["tokenizer"] = tokenizer
        # Extract the inner base model — attribute name varies by architecture
        if hasattr(model, "model"):           # Llama, Qwen, Mistral
            model_state["transformer"] = model.model
        elif hasattr(model, "transformer"):   # GPT-2, GPT-Neo
            model_state["transformer"] = model.transformer
        elif hasattr(model, "gpt_neox"):      # GPT-NeoX, Pythia
            model_state["transformer"] = model.gpt_neox
        else:
            model_state["transformer"] = model
        model_state["model"] = None
        model_state["causal_lm"] = model
        # Store final layer norm and lm_head for projecting hidden states to vocab
        causal_lm = model
        if hasattr(causal_lm, "transformer") and hasattr(causal_lm.transformer, "ln_f"):
            model_state["final_norm"] = causal_lm.transformer.ln_f      # GPT-2, GPT-Neo
        elif hasattr(causal_lm, "model") and hasattr(causal_lm.model, "norm"):
            model_state["final_norm"] = causal_lm.model.norm            # Llama, Mistral
        elif hasattr(causal_lm, "gpt_neox") and hasattr(causal_lm.gpt_neox, "final_layer_norm"):
            model_state["final_norm"] = causal_lm.gpt_neox.final_layer_norm  # GPT-NeoX
        else:
            model_state["final_norm"] = None
        model_state["lm_head"] = causal_lm.lm_head
    else:
        sentence_transformer = SentenceTransformer(model_name)
        model_state["model"] = sentence_transformer
        model_state["tokenizer"] = sentence_transformer.tokenizer
        model_state["transformer"] = sentence_transformer[0].auto_model
        model_state["causal_lm"] = None

    # Step 2 — Build the vocabulary
    _update_progress(2, "Building vocabulary", "Extracting from tokenizer...")
    print("\n[2/3] Building vocabulary...")
    model_state["vocab_words"] = _load_vocab(model_state["tokenizer"])
    vocab_words = model_state["vocab_words"]

    # Precompute token-ID lists for fast logit→vocab mapping (generative only)
    if model_type == "generative":
        model_state["vocab_token_ids"] = [
            model_state["tokenizer"].encode(w, add_special_tokens=False)
            for w in vocab_words
        ]
    else:
        model_state["vocab_token_ids"] = None

    # Step 3 — Load or compute vocabulary embeddings
    _update_progress(3, "Loading embeddings", "Checking cache...")
    print("\n[3/3] Loading vocabulary embeddings...")
    cache = _vocab_cache_path(model_name)
    if cache.exists():
        vocab_embs = np.load(cache)
        if vocab_embs.shape[0] != len(vocab_words):
            print(f"Cache stale — re-encoding {len(vocab_words)} words")
            vocab_embs = _encode_vocab()
            np.save(cache, vocab_embs)
        else:
            _update_progress(3, "Loading embeddings", f"Loaded from cache ({len(vocab_words)} words)", 100)
            print(f"Loaded from cache ({len(vocab_words)} words)")
    else:
        print(f"Encoding {len(vocab_words)} words (first time, will be cached)")
        vocab_embs = _encode_vocab()
        np.save(cache, vocab_embs)
    model_state["vocab_embeddings"] = vocab_embs

    print(f"\nReady — {model_name}\n")
