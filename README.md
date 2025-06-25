# slideSimCheck

This repository explores the distribution of semantic signal across the dimensions of sentence embeddings. It implements two main strategies to identify high-signal subregions in standard dense embeddings, using similarity tasks as a proxy for semantic quality.

The core idea is to analyze whether certain embedding regions are more informative than others, even in models not trained with progressive objectives (e.g., Matryoshka Representation Learning).

---

## Overview

Two analysis modes are implemented:

* **Sliding Window**: Measure performance using fixed-size, moving slices of the embedding vector.
* **Region Growing**: Start from high-performing slices and grow them incrementally, retaining only performance-improving expansions.

All experiments evaluate embedding slices on similarity tasks from the [MTEB benchmark](https://github.com/embeddings-benchmark/mteb), using cosine similarity and Spearman correlation with gold labels.

---

## Installation

```bash
uv sync
```

---

## Usage

All analyses are run via `main.py`. The key arguments are:

* `--model_name`: SentenceTransformer model (e.g., `tomaarsen/mpnet-base-nli`)
* `--task_name`: MTEB similarity task (e.g., `STSBenchmark`)
* `--analysis_mode`: `sliding_window`, `multi_split_sliding_window`, or `region_growing`

### Example 1: Sliding Window

```bash
python main.py \
  --analysis_mode sliding_window \
  --model_name tomaarsen/mpnet-base-nli \
  --task_name STSBenchmark \
  --window_size 64 \
  --step_size 16
```

### Example 2: Multi-Split Sliding Window

```bash
python main.py \
  --analysis_mode multi_split_sliding_window \
  --model_name tomaarsen/mpnet-base-nli \
  --task_name STSBenchmark \
  --window_size 64 \
  --step_size 16
```

### Example 3: Region Growing

```bash
python main.py \
  --analysis_mode region_growing \
  --model_name tomaarsen/mpnet-base-nli \
  --task_name STSBenchmark \
  --initial_step 16 \
  --growth_increment 8 \
  --num_seeds 3
```

---

## Outputs

Each run saves plots to the root directory. Examples include:

* `sliding_window_summary.png`: Spearman score across embedding slices
* `split_trend_comparison_plot.png`: Train/val/test performance per slice
* `performance_distribution_plot.png`: Normalized signal intensity by dimension
* `multi_seed_region_growing_analysis.png`: Region expansion traces from each seed

---

## File Structure

* `main.py`: Orchestrates analysis from CLI
* `utils.py`: Embedding slicing, scoring, plotting
* `SmartPairWrapper`: MTEB-compatible wrapper using direct cosine similarity
* `PrecomputedEmbeddingWrapper`: Precomputes embeddings and handles dataset splits

---

## Context

This codebase complements Matryoshka Representation Learning (MRL) by analyzing dimension-level signal in standard models. Instead of learning to sort dimensions by utility, we evaluate signal distribution directly via selective masking and ablation.

---

## License

MIT
