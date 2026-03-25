"""Builds the JSON payload that the Three.js frontend renders."""

import numpy as np


def _viridis_color(value, value_min, value_max):
    """Map a value to a Viridis CSS rgb() string and a 0–1 fraction."""
    t = (value - value_min) / (value_max - value_min) if value_max > value_min else 1.0
    stops = [(68, 1, 84), (49, 104, 142), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    s = t * (len(stops) - 1)
    idx = min(int(s), len(stops) - 2)
    frac = s - idx
    r = int(stops[idx][0] + frac * (stops[idx + 1][0] - stops[idx][0]))
    g = int(stops[idx][1] + frac * (stops[idx + 1][1] - stops[idx][1]))
    b = int(stops[idx][2] + frac * (stops[idx + 1][2] - stops[idx][2]))
    return f"rgb({r},{g},{b})", t


def _lava_color(value, value_min, value_max):
    """Map a value to a lava-palette CSS rgb() string and a 0–1 fraction."""
    t = (value - value_min) / (value_max - value_min) if value_max > value_min else 1.0
    stops = [(48, 8, 8), (140, 20, 10), (210, 60, 10), (240, 140, 20), (255, 220, 80)]
    s = t * (len(stops) - 1)
    idx = min(int(s), len(stops) - 2)
    frac = s - idx
    r = int(stops[idx][0] + frac * (stops[idx + 1][0] - stops[idx][0]))
    g = int(stops[idx][1] + frac * (stops[idx + 1][1] - stops[idx][1]))
    b = int(stops[idx][2] + frac * (stops[idx + 1][2] - stops[idx][2]))
    return f"rgb({r},{g},{b})", t


def _make_point(word, coords, n_dims, **extra):
    """Build a single point dict for the scene JSON."""
    point = {
        "word": word,
        "x": float(coords[0]),
        "y": float(coords[1]),
        "z": float(coords[2]) if n_dims == 3 else 0.0,
    }
    point.update(extra)
    return point


def build_scene_json(
    sentence_coords, neighbor_coords, words, neighbor_words,
    clicked_idx, neighbor_sims, is_3d,
    isolated_coords=None, candidate_coords=None,
    candidate_words=None, candidate_probs=None,
    iso_neighbor_coords=None, iso_neighbor_words=None, iso_neighbor_sims=None,
):
    """Create the JSON payload consumed by the Three.js scene builder."""
    # Normalize all coordinates to a fixed scale centered around the mean
    all_coords = np.vstack([sentence_coords, neighbor_coords])
    center = all_coords.mean(axis=0)
    extent = max(np.abs(all_coords - center).max(), 1e-6)
    scale = 3.0 / extent

    sentence_norm = (sentence_coords - center) * scale
    neighbor_norm = (neighbor_coords - center) * scale
    n_dims = 3 if is_3d else 2

    # Normalize similarities to 0–1 for coloring
    if neighbor_sims:
        sim_min, sim_max = min(neighbor_sims), max(neighbor_sims)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
        norm_sims = [(s - sim_min) / sim_range for s in neighbor_sims]
    else:
        norm_sims = [0.5] * len(neighbor_words)

    # Selected word
    sentence_points = [
        _make_point(words[clicked_idx], sentence_norm[clicked_idx], n_dims, selected=True),
    ]

    # Neighbor points
    neighbor_points = [
        _make_point(w, neighbor_norm[i], n_dims, tint=float(norm_sims[i]), rawSim=float(neighbor_sims[i]))
        for i, w in enumerate(neighbor_words)
    ]

    # Isolated-embedding point (word encoded without sentence context)
    isolated_point = None
    if isolated_coords is not None:
        iso_norm = (isolated_coords[0] - center) * scale
        isolated_point = _make_point(words[clicked_idx] + " (isolated)", iso_norm, n_dims)

    # Isolated-embedding neighbor points
    iso_neighbor_points = []
    if iso_neighbor_coords is not None and iso_neighbor_words:
        iso_n_norm = (iso_neighbor_coords - center) * scale
        if iso_neighbor_sims:
            iso_sim_min, iso_sim_max = min(iso_neighbor_sims), max(iso_neighbor_sims)
            iso_sim_range = iso_sim_max - iso_sim_min if iso_sim_max > iso_sim_min else 1.0
            iso_norm_sims = [(s - iso_sim_min) / iso_sim_range for s in iso_neighbor_sims]
        else:
            iso_norm_sims = [0.5] * len(iso_neighbor_words)
        for i, w in enumerate(iso_neighbor_words):
            iso_neighbor_points.append(_make_point(
                w, iso_n_norm[i], n_dims,
                tint=float(iso_norm_sims[i]), rawSim=float(iso_neighbor_sims[i]),
            ))

    # Next-token candidate points (generative models only)
    candidate_points = []
    if candidate_coords is not None and candidate_words:
        cand_norm = (candidate_coords - center) * scale
        prob_max = max(candidate_probs) if candidate_probs else 1.0
        for i, w in enumerate(candidate_words):
            candidate_points.append(_make_point(
                w, cand_norm[i], n_dims,
                prob=float(candidate_probs[i]),
                tint=float(candidate_probs[i] / prob_max) if prob_max > 0 else 0.5,
            ))

    return {
        "mode": "3d" if is_3d else "2d",
        "sentencePoints": sentence_points,
        "neighborPoints": neighbor_points,
        "selectedIndex": 0,
        "isolatedPoint": isolated_point,
        "isoNeighborPoints": iso_neighbor_points,
        "candidatePoints": candidate_points,
    }


def build_neighbors_panel(neighbor_words, sims, top_indices, n_neighbors, color_fn=None):
    """Build the neighbor list for the inspector sidebar (top 15)."""
    if color_fn is None:
        color_fn = _viridis_color
    top_n = min(15, n_neighbors)
    top_sims = [float(sims[i]) for i in top_indices[:top_n]]
    sim_min = min(top_sims) if top_sims else 0
    sim_max = max(top_sims) if top_sims else 1
    neighbors = []
    for i in range(top_n):
        color, fraction = color_fn(top_sims[i], sim_min, sim_max)
        neighbors.append({
            "word": neighbor_words[i],
            "sim": top_sims[i],
            "bar_color": color,
            "bar_t": fraction,
        })
    return neighbors
