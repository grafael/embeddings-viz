# Embedding Projector

Interactive 3D visualization of word embeddings in semantic space. Type a sentence, click a word, and explore how it relates to its nearest neighbors in real time.

Supports both **embedding models** (bidirectional) and **generative models** (causal), with layer selection and next-token prediction.

## Features

- **Contextual embeddings** — words encoded using full sentence context
- **3D/2D projection** — UMAP, t-SNE, or PCA dimensionality reduction
- **Nearest neighbors** — closest vocabulary words by cosine similarity
- **Layer selection** — extract embeddings from any transformer layer
- **Isolated vs. contextual** — compare how sentence context shifts a word's embedding
- **Next-token prediction** — top candidates with probabilities (generative models)
- **Model switching** — swap models on the fly from the UI

## Getting Started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/grafael/embeddings-viz.git
cd embeddings-viz
uv sync
uv run embeddings-viz
```

Open http://localhost:8050.

## Configuration

Edit `config.yaml` to set the default model and available models:

```yaml
default_model: all-MiniLM-L6-v2

models:
  - all-MiniLM-L6-v2
  - all-mpnet-base-v2
  - name: gpt2
    type: generative
```

Vocabulary is extracted automatically from each model's tokenizer. To override it, place a `vocab.txt` file (one word per line) in the project root. Vocabulary embeddings are cached in `vocab/` on first run.

## Supported Models

| Model | Type | Params | Layers |
|-------|------|--------|--------|
| all-MiniLM-L6-v2 | Embedding | 22M | 6 |
| all-MiniLM-L12-v2 | Embedding | 33M | 12 |
| all-mpnet-base-v2 | Embedding | 109M | 12 |
| all-distilroberta-v1 | Embedding | 82M | 6 |
| paraphrase-MiniLM-L6-v2 | Embedding | 22M | 6 |
| gpt2 | Generative | 124M | 12 |
| distilgpt2 | Generative | 82M | 6 |
| Qwen/Qwen2.5-0.5B | Generative | 500M | 24 |
| HuggingFaceTB/SmolLM2-135M | Generative | 135M | 30 |
| openai-community/gpt2-medium | Generative | 355M | 24 |

Additional models can be added in `config.yaml`.

## License

MIT
