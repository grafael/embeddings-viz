"""Flask routes and request handling."""

import re
import threading

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

from embeddings_viz.config import AVAILABLE_MODELS, DEFAULT_MODEL, MODEL_NAMES
from embeddings_viz.embeddings import (
    HAS_UMAP,
    _find_word_spans,
    encode_single_word,
    get_contextual_word_embeddings,
    reduce_dimensions,
)
from embeddings_viz.models import load_model, model_state, progress
from embeddings_viz.scene import build_neighbors_panel, build_scene_json

app = Flask(__name__)
session_state = {"sentence": "", "words": [], "selected_idx": 0}



def _find_neighbors(selected_emb, selected_word, n_neighbors):
    """Find the nearest vocabulary neighbors for a given embedding."""
    vocab_words = model_state["vocab_words"]
    sims = cosine_similarity(selected_emb, model_state["vocab_embeddings"])[0]
    ranked = np.argsort(sims)[::-1]
    top_indices = np.array([
        i for i in ranked if vocab_words[i].lower() != selected_word.lower()
    ][:n_neighbors])
    neighbor_words = [vocab_words[i] for i in top_indices]
    neighbor_embs = model_state["vocab_embeddings"][top_indices]
    neighbor_sims = [float(sims[i]) for i in top_indices]
    return sims, top_indices, neighbor_words, neighbor_embs, neighbor_sims


def _get_next_token_candidates(sentence, words, selected_idx, layer, temperature, n_candidates=15):
    """Compute next-token predictions for generative models.

    Returns (candidate_words, candidate_probs, candidate_embs, predicted_token, predicted_rank).
    For embedding models returns empty/None values.
    """
    if model_state["type"] != "generative" or model_state["causal_lm"] is None:
        return [], [], None, None, None

    tokenizer = model_state["tokenizer"]
    encoded = tokenizer(
        sentence, return_tensors="pt", truncation=True, max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    # Find the last token that overlaps the selected word
    word_spans = _find_word_spans(sentence, words)
    word_start, word_end = word_spans[selected_idx]
    last_tok_idx = 0
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue
        if max(tok_start, word_start) < min(tok_end, word_end):
            last_tok_idx = tok_idx

    device = next(model_state["causal_lm"].parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model_state["causal_lm"](**encoded).logits
    next_logits = logits[0, last_tok_idx, :]

    # Top-K candidates with probabilities
    raw_probs = torch.softmax(next_logits, dim=-1)
    top_k = torch.topk(raw_probs, n_candidates)
    raw_words = [tokenizer.decode([tid]).strip() for tid in top_k.indices.tolist()]
    raw_probs_list = top_k.values.tolist()
    filtered = [(w, p) for w, p in zip(raw_words, raw_probs_list) if w]
    candidate_words = [w for w, _ in filtered]
    candidate_probs = [p for _, p in filtered]

    # Compute embeddings for each candidate word
    candidate_embs = (
        np.array([encode_single_word(w, layer) for w in candidate_words])
        if candidate_words else None
    )

    # Sample or argmax a predicted token (with temperature)
    if temperature > 0:
        temp_probs = torch.softmax(next_logits / temperature, dim=-1)
        predicted_id = torch.multinomial(temp_probs, 1).item()
    else:
        predicted_id = next_logits.argmax().item()
    predicted_token = tokenizer.decode([predicted_id]).strip()
    predicted_rank = int((raw_probs >= raw_probs[predicted_id]).sum().item())

    return candidate_words, candidate_probs, candidate_embs, predicted_token, predicted_rank


def _slice_coordinates(coords, n_words, has_isolated, candidate_words, has_candidates):
    """Slice the combined coordinate array back into per-group arrays.

    The concatenation order must match what was passed to reduce_dimensions:
      [sentence_embs, isolated_emb?, candidate_embs?, neighbor_embs]
    """
    idx = 0

    sentence_coords = coords[idx : idx + n_words]
    idx += n_words

    isolated_coords = None
    if has_isolated:
        isolated_coords = coords[idx : idx + 1]
        idx += 1

    candidate_coords = None
    if has_candidates:
        candidate_coords = coords[idx : idx + len(candidate_words)]
        idx += len(candidate_words)

    neighbor_coords = coords[idx:]
    return sentence_coords, isolated_coords, candidate_coords, neighbor_coords



@app.route("/")
def index():
    emb_models = [m["name"] for m in AVAILABLE_MODELS if m["type"] == "embedding"]
    gen_models = [m["name"] for m in AVAILABLE_MODELS if m["type"] == "generative"]
    return render_template(
        "index.html",
        has_umap=HAS_UMAP,
        emb_models=emb_models,
        gen_models=gen_models,
        current_model=model_state["name"],
        current_type=model_state.get("type", "embedding"),
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Tokenize a sentence into words and return metadata."""
    data = request.json
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "Empty sentence"}), 400
    words = re.findall(r"\b\w+\b", sentence)
    if not words:
        return jsonify({"error": "No words found"}), 400
    session_state["sentence"] = sentence
    session_state["words"] = words
    session_state["selected_idx"] = 0
    return jsonify({
        "words": words,
        "selected_idx": 0,
        "num_layers": model_state["transformer"].config.num_hidden_layers,
        "model_type": model_state["type"],
    })


@app.route("/api/visualize", methods=["POST"])
def visualize():
    """Compute embeddings, neighbors, and projected coordinates for the selected word."""
    data = request.json
    selected_idx = data.get("selected_idx", 0)
    dr_method = data.get("dr_method", "pca")
    n_dims = data.get("n_dims", 3)
    n_neighbors = data.get("n_neighbors", 30)
    show_isolated = data.get("show_isolated", False)
    layer = data.get("layer", None)
    temperature = data.get("temperature", 1.0)

    words = session_state["words"]
    sentence = session_state["sentence"]
    if not words or not sentence:
        return jsonify({"error": "No sentence analyzed"}), 400

    selected_word = words[selected_idx]
    sentence_embs = get_contextual_word_embeddings(sentence, words, layer=layer)
    selected_emb = sentence_embs[selected_idx].reshape(1, -1)

    # Nearest vocabulary neighbors
    sims, top_indices, neighbor_words, neighbor_embs, neighbor_sims = _find_neighbors(
        selected_emb, selected_word, n_neighbors,
    )

    # Isolated embedding (same word without sentence context)
    isolated_emb = None
    if show_isolated:
        isolated_emb = get_contextual_word_embeddings(selected_word, [selected_word], layer=layer)

    # Next-token candidates (generative models only)
    candidate_words, candidate_probs, candidate_embs, predicted_token, predicted_rank = (
        _get_next_token_candidates(sentence, words, selected_idx, layer, temperature)
    )

    # Reduce all embeddings together so they share the same coordinate space
    emb_parts = [sentence_embs]
    if isolated_emb is not None:
        emb_parts.append(isolated_emb)
    if candidate_embs is not None:
        emb_parts.append(candidate_embs)
    emb_parts.append(neighbor_embs)
    coords = reduce_dimensions(np.vstack(emb_parts), dr_method, n_dims)

    sentence_coords, isolated_coords, cand_coords, neighbor_coords = _slice_coordinates(
        coords, len(words),
        isolated_emb is not None, candidate_words, candidate_embs is not None,
    )

    scene = build_scene_json(
        sentence_coords, neighbor_coords, words, neighbor_words,
        selected_idx, neighbor_sims, n_dims == 3,
        isolated_coords=isolated_coords,
        candidate_coords=cand_coords,
        candidate_words=candidate_words,
        candidate_probs=candidate_probs,
    )

    neighbors_panel = build_neighbors_panel(neighbor_words, sims, top_indices, n_neighbors)

    result = {
        "selected_word": selected_word,
        "neighbors": neighbors_panel,
        "scene": scene,
        "embedding": selected_emb.flatten().tolist(),
    }
    if isolated_emb is not None:
        result["isolated_embedding"] = isolated_emb.flatten().tolist()
    if predicted_token is not None:
        result["predicted_token"] = predicted_token
        result["predicted_rank"] = predicted_rank
    if candidate_words:
        result["candidates"] = [
            {"word": w, "prob": p} for w, p in zip(candidate_words, candidate_probs)
        ]

    return jsonify(result)



def _do_load_model(model_name):
    """Background thread target for model loading."""
    progress["active"] = True
    progress["done"] = False
    progress["error"] = None
    progress["result"] = None
    try:
        load_model(model_name)
        session_state["sentence"] = ""
        session_state["words"] = []
        session_state["selected_idx"] = 0
        progress["result"] = {
            "status": "ok",
            "model": model_name,
            "num_layers": model_state["transformer"].config.num_hidden_layers,
            "model_type": model_state["type"],
        }
    except Exception as e:
        progress["error"] = str(e)
    finally:
        progress["active"] = False
        progress["done"] = True


@app.route("/api/switch_model", methods=["POST"])
def switch_model():
    data = request.json
    model_name = data.get("model", "").strip()
    if model_name not in MODEL_NAMES:
        return jsonify({"error": "Unknown model"}), 400
    if model_name == model_state["name"]:
        return jsonify({"status": "ok", "model": model_name})
    if progress["active"]:
        return jsonify({"error": "Model loading already in progress"}), 409
    thread = threading.Thread(target=_do_load_model, args=(model_name,))
    thread.start()
    return jsonify({"status": "loading"})


@app.route("/api/progress")
def get_progress():
    return jsonify(progress)



def main():
    load_model(DEFAULT_MODEL)
    app.run(debug=False, host="0.0.0.0", port=8050, threaded=True)
