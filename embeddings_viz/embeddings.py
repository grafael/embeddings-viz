"""Word-level embedding extraction and dimensionality reduction."""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embeddings_viz.models import model_state

try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False



def _find_word_spans(sentence, words):
    """Return (start, end) character offsets for each word in order."""
    spans = []
    search_start = 0
    sentence_lower = sentence.lower()
    for word in words:
        idx = sentence_lower.find(word.lower(), search_start)
        if idx == -1:
            idx = sentence_lower.find(word.lower())
        spans.append((idx, idx + len(word)))
        search_start = idx + len(word)
    return spans


def encode_single_word(word, layer=None):
    """Encode a single word in isolation — works for both model types."""
    if model_state["type"] == "embedding" and model_state["model"] is not None:
        return model_state["model"].encode([word])[0]
    device = model_state["transformer"].device
    encoded = model_state["tokenizer"](word, return_tensors="pt", truncation=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model_state["transformer"](**encoded, output_hidden_states=True)
    if layer is not None and output.hidden_states:
        hidden = output.hidden_states[layer].squeeze(0)
    else:
        hidden = output.last_hidden_state.squeeze(0)
    return hidden.mean(dim=0).cpu().numpy()


def get_contextual_word_embeddings(sentence, words, layer=None):
    """Embed each word using its full sentence context, averaging over sub-tokens."""
    device = model_state["transformer"].device
    encoded = model_state["tokenizer"](
        sentence, return_tensors="pt", truncation=True, max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model_state["transformer"](**encoded, output_hidden_states=True)

    if layer is not None and output.hidden_states is not None:
        token_embeddings = output.hidden_states[layer].squeeze(0).cpu().numpy()
    else:
        token_embeddings = output.last_hidden_state.squeeze(0).cpu().numpy()

    word_spans = _find_word_spans(sentence, words)
    word_embeddings = []
    for word, (word_start, word_end) in zip(words, word_spans):
        # Collect sub-token indices that overlap this word's character span
        subword_indices = [
            tok_idx
            for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping)
            if tok_start != tok_end and max(tok_start, word_start) < min(tok_end, word_end)
        ]
        if subword_indices:
            word_embeddings.append(token_embeddings[subword_indices].mean(axis=0))
        else:
            word_embeddings.append(encode_single_word(word, layer))
    return np.array(word_embeddings)



def get_all_layer_embeddings(sentence, words):
    """Get word embeddings at every hidden layer in a single forward pass."""
    device = model_state["transformer"].device
    encoded = model_state["tokenizer"](
        sentence, return_tensors="pt", truncation=True, max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model_state["transformer"](**encoded, output_hidden_states=True)

    word_spans = _find_word_spans(sentence, words)
    all_layers = []
    for hidden_state in output.hidden_states:
        token_embs = hidden_state.squeeze(0).cpu().numpy()
        word_embs = []
        for word, (ws, we) in zip(words, word_spans):
            indices = [
                t for t, (ts, te) in enumerate(offset_mapping)
                if ts != te and max(ts, ws) < min(te, we)
            ]
            if indices:
                word_embs.append(token_embs[indices].mean(axis=0))
            else:
                word_embs.append(encode_single_word(word))
        all_layers.append(np.array(word_embs))
    return all_layers


def reduce_dimensions(embeddings, method, n_components):
    """Project high-dimensional embeddings to 2D or 3D."""
    n = min(n_components, embeddings.shape[0], embeddings.shape[1])
    if method == "pca":
        coords = PCA(n_components=n).fit_transform(embeddings)
    elif method == "tsne":
        perplexity = min(30, max(2, embeddings.shape[0] - 1))
        coords = TSNE(
            n_components=n, perplexity=perplexity, random_state=42, max_iter=800,
        ).fit_transform(embeddings)
    elif method == "umap" and HAS_UMAP:
        n_neighbors = min(15, embeddings.shape[0] - 1)
        coords = UMAP(
            n_components=n, n_neighbors=n_neighbors, random_state=42, n_jobs=1,
        ).fit_transform(embeddings)
    else:
        coords = PCA(n_components=n).fit_transform(embeddings)
    # Pad to the requested number of dimensions if needed
    if coords.shape[1] < n_components:
        padding = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, padding])
    return coords
