import torch
import mteb
from mteb import MTEB
from mteb.abstasks import (
    AbsTaskClassification,
    AbsTaskClustering,
    AbsTaskPairClassification,
    AbsTaskReranking,
    AbsTaskSTS,
)
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse
import logging

# --- Suppress MTEB Metadata Warning ---
# The MTEB library issues a warning when it cannot find model card metadata.
# This is expected when using our custom wrapper classes (SmartPairWrapper, 
# PrecomputedEmbeddingWrapper) because they are not full SentenceTransformer
# models. The warning is purely informational and does not affect the evaluation
# results, so we can safely suppress it to keep the logs clean.
logging.getLogger("mteb.models.overview").setLevel(logging.ERROR)

# --- 2. EFFICIENT HELPER CLASSES AND EVALUATION LOGIC ---

class PrecomputedEmbeddingWrapper:
    """
    A wrapper that mimics a SentenceTransformer's .encode() method but uses
    pre-computed embeddings and simply returns a slice of them.
    """
    def __init__(self, full_embeddings: np.ndarray, start_dim: int, end_dim: int):
        self.sliced_embeddings = full_embeddings[:, start_dim:end_dim]

    def encode(self, sentences: List[str], *args, **kwargs) -> np.ndarray:
        return self.sliced_embeddings[: len(sentences)]

class SmartPairWrapper:
    """
    A wrapper for sentence-pair tasks (e.g., STS) that uses a pre-computed
    mapping of sentences to their embeddings.
    """
    def __init__(self, sentence_map: Dict[str, np.ndarray]):
        self.sentence_map = sentence_map

    def encode(self, sentences: List[str], *args, **kwargs) -> np.ndarray:
        return np.array([self.sentence_map[s] for s in sentences])

def create_precomputed_embeddings(task: mteb.AbsTask, model: SentenceTransformer, batch_size: int) -> Dict[str, Any]:
    """
    Analyzes the task type and creates pre-computed embeddings for all splits.
    """
    print(f"INFO: Analyzing task '{task.metadata_dict['name']}' of type {task.__class__.__name__}")
    precomputed_embeddings = {}
    is_pair_task = isinstance(task, (AbsTaskSTS, AbsTaskPairClassification))

    # --- FIX IS HERE: Iterate over all dataset keys, not just eval_splits ---
    for split in task.dataset.keys():
        print(f"INFO: Pre-computing embeddings for split: '{split}'")
        dataset = task.dataset[split]
        if is_pair_task:
            sentences1 = dataset["sentence1"]
            sentences2 = dataset["sentence2"]
            corpus = list(set(sentences1 + sentences2))
            encoded_corpus = model.encode(corpus, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
            sent_to_emb = {sent: emb for sent, emb in zip(corpus, encoded_corpus)}
            embeddings1 = np.array([sent_to_emb[s] for s in sentences1])
            embeddings2 = np.array([sent_to_emb[s] for s in sentences2])
            precomputed_embeddings[split] = {"sents1": sentences1, "embs1": embeddings1, "sents2": sentences2, "embs2": embeddings2}
        else:
            corpus = dataset['text'] if 'text' in dataset.features else dataset['sentences']
            embeddings = model.encode(corpus, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
            precomputed_embeddings[split] = embeddings
    return precomputed_embeddings

def evaluate_single_slice_parallel(task_name: str, metric: str, all_embeddings: Dict[str, Any], dims: Tuple[int, int], eval_split: str) -> Tuple[int, int, float, str]:
    """
    Evaluates a single slice using pre-computed embeddings for a given split.
    """
    start_dim, end_dim = dims
    task = mteb.get_tasks(tasks=[task_name])[0]
    split_data = all_embeddings[eval_split]
    evaluator = MTEB(tasks=[task])
    is_pair_structure = isinstance(split_data, dict) and 'sents1' in split_data

    if is_pair_structure:
        sliced_embs1 = split_data['embs1'][:, start_dim:end_dim]
        sliced_embs2 = split_data['embs2'][:, start_dim:end_dim]
        sentence_map = {s: e for s, e in zip(split_data['sents1'], sliced_embs1)}
        sentence_map.update({s: e for s, e in zip(split_data['sents2'], sliced_embs2)})
        wrapper = SmartPairWrapper(sentence_map)
    else:
        wrapper = PrecomputedEmbeddingWrapper(split_data, start_dim, end_dim)

    results = evaluator.run(wrapper, output_folder=None, eval_splits=[eval_split])
    try:
        score = results[0].scores[eval_split][0][metric] * 100
        return start_dim, end_dim, score, eval_split
    except (KeyError, IndexError, TypeError):
        return start_dim, end_dim, 0.0, eval_split

def get_task_metric(task_object: Any) -> str:
    """Determines the appropriate MTEB metric based on the task type."""
    if isinstance(task_object, AbsTaskSTS): return "spearman"
    if isinstance(task_object, AbsTaskClassification): return "accuracy"
    if isinstance(task_object, AbsTaskClustering): return "v_measure"
    if isinstance(task_object, AbsTaskReranking): return "map"
    if isinstance(task_object, AbsTaskPairClassification): return "accuracy"
    return "main_score"

# --- 3. ANALYSIS ALGORITHMS ---

def run_sliding_window_search(task_name: str, metric: str, all_embeddings: Dict, params: Dict, total_dims: int, max_workers: int) -> pd.DataFrame:
    """
    Performs a sliding window search using pre-computed embeddings.
    """
    print("\n--- Running Analysis: Sliding Window ---")
    window_size, step_size = params["window_size"], params["step_size"]
    dims_to_evaluate = [(s, s + window_size) for s in range(0, total_dims - window_size + 1, step_size)]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_slice_parallel, task_name, metric, all_embeddings, dims, 'test') for dims in dims_to_evaluate]
        for future in as_completed(futures):
            start_dim, end_dim, score, _ = future.result()
            results.append({"window": f"{start_dim}-{end_dim}", "score": score})
    return pd.DataFrame(results).sort_values(by="score", ascending=False)

def run_multi_split_sliding_window_search(task_name: str, metric: str, all_embeddings: Dict, params: Dict, total_dims: int, max_workers: int) -> pd.DataFrame:
    """
    Performs a sliding window search across all data splits (train, validation, test).
    """
    print("\n--- Running Analysis: Multi-Split Sliding Window ---")
    window_size, step_size = params["window_size"], params["step_size"]
    dims_to_evaluate = [(s, s + window_size) for s in range(0, total_dims - window_size + 1, step_size)]
    tasks = []
    splits = all_embeddings.keys()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for dims in dims_to_evaluate:
            for split in splits:
                tasks.append(executor.submit(evaluate_single_slice_parallel, task_name, metric, all_embeddings, dims, split))
        
        results_map = {}
        for future in as_completed(tasks):
            start_dim, end_dim, score, split = future.result()
            window = f"{start_dim}-{end_dim}"
            if window not in results_map: results_map[window] = {}
            results_map[window][f"{split}_score"] = score

    results_list = [{"window": window, **scores} for window, scores in results_map.items()]
    return pd.DataFrame(results_list)

def grow_single_seed_worker(task_name: str, metric: str, all_embeddings: Dict, seed_dims: Tuple[int, int], seed_score: float, growth_increment: int, total_dims: int) -> Dict:
    """Worker function to grow a single seed region."""
    current_start, current_end = seed_dims
    current_best_score = seed_score

    def fast_evaluate(dims):
        _, _, score, _ = evaluate_single_slice_parallel(task_name, metric, all_embeddings, dims, 'test')
        return score

    while True:
        grew = False
        # Try growing right
        right_dims = (current_start, min(current_end + growth_increment, total_dims))
        if right_dims[1] > current_end:
            right_score = fast_evaluate(right_dims)
            if right_score > current_best_score:
                current_end, current_best_score = right_dims[1], right_score
                grew = True
        # Try growing left
        left_dims = (max(current_start - growth_increment, 0), current_end)
        if left_dims[0] < current_start:
            left_score = fast_evaluate(left_dims)
            if left_score > current_best_score:
                current_start, current_best_score = left_dims[0], left_score
                grew = True
        if not grew:
            break
    return {"seed_window": f"{seed_dims[0]}-{seed_dims[1]}", "final_window": f"{current_start}-{current_end}", "final_score": current_best_score}

def run_multi_seed_region_growing_search(task_name: str, metric: str, all_embeddings: Dict, params: Dict, total_dims: int, max_workers: int) -> pd.DataFrame:
    """Performs the multi-seed region growing search."""
    print("\n--- Running Analysis: Multi-Seed Region Growing ---")
    initial_step, growth_increment, num_seeds = params["initial_step"], params["growth_increment"], params["num_seeds"]
    dims_to_evaluate = [(s, min(s + initial_step, total_dims)) for s in range(0, total_dims, initial_step)]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_slice_parallel, task_name, metric, all_embeddings, dims, 'test') for dims in dims_to_evaluate]
        initial_scan_results = [{"window": f"{r[0]}-{r[1]}", "score": r[2]} for r in [f.result() for f in as_completed(futures)]]

    scan_df = pd.DataFrame(initial_scan_results).sort_values(by="score", ascending=False)
    top_seeds_df = scan_df.head(num_seeds)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        growth_futures = [executor.submit(grow_single_seed_worker, task_name, metric, all_embeddings, tuple(map(int, row["window"].split("-"))), row["score"], growth_increment, total_dims) for _, row in top_seeds_df.iterrows()]
        final_hotspots = [f.result() for f in as_completed(growth_futures)]

    # Prepare DataFrame for plotting
    plot_df = scan_df.assign(type="Initial Scan")
    for hotspot in final_hotspots:
        plot_df.loc[plot_df["window"] == hotspot["seed_window"], "type"] = "Search Seed"
    hotspot_data = [{"window": h["final_window"], "score": h["final_score"], "type": "Grown Hotspot"} for h in final_hotspots]
    plot_df = pd.concat([plot_df, pd.DataFrame(hotspot_data)], ignore_index=True)
    if not plot_df[plot_df["type"] == "Grown Hotspot"].empty:
        best_idx = plot_df[plot_df["type"] == "Grown Hotspot"]["score"].idxmax()
        plot_df.loc[best_idx, "type"] = "Best Overall Hotspot"
    return plot_df

# --- 4. VISUALIZATION FUNCTIONS ---

def plot_sliding_window_summary(df: pd.DataFrame, model_name: str, task_name: str):
    """Plots the results of the simple sliding window search."""
    plt.figure(figsize=(16, 8))
    sns.barplot(x="window", y="score", data=df, palette="viridis")
    plt.title(f"Performance of Embedding Segments on {task_name}\nModel: {model_name}", fontsize=16)
    plt.xlabel("Embedding Dimension Window", fontsize=12)
    plt.ylabel(f"Performance ({'Spearman' if 'STS' in task_name else 'Score'} %)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("sliding_window_summary.png")
    plt.show()

def analyze_and_visualize_distribution(df: pd.DataFrame):
    """Calculates statistics and plots the normalized performance distribution."""
    if df.empty: return
    df[['start', 'end']] = df['window'].str.split('-', expand=True).astype(int)
    df['center_dim'] = (df['start'] + df['end']) / 2
    min_score, max_score = df['score'].min(), df['score'].max()
    df['normalized_score'] = (df['score'] - min_score) / (max_score - min_score)
    df = df.sort_values('center_dim')

    plt.figure(figsize=(14, 7))
    sns.lineplot(x='center_dim', y='normalized_score', data=df, marker='o', label='Normalized Performance')
    plt.fill_between(df['center_dim'], df['normalized_score'], alpha=0.2)
    plt.title("Normalized Performance Distribution", fontsize=18)
    plt.xlabel("Center of Dimension Window", fontsize=12)
    plt.ylabel("Normalized Performance", fontsize=12)
    plt.savefig("performance_distribution_plot.png")
    plt.show()

def analyze_and_plot_split_comparison(df: pd.DataFrame):
    """Analyzes and plots performance trends across data splits."""
    if df.empty: return
    df_clean = df.dropna()
    
    # Check for required columns before using them
    required_cols = ['test_score', 'validation_score', 'train_score']
    if not all(col in df_clean.columns for col in required_cols):
        print("DataFrame is missing one or more required score columns for correlation analysis.")
        print(f"Available columns: {df_clean.columns.tolist()}")
        return

    print("Correlation Matrix:\n", df_clean[required_cols].corr())
    
    df['center_dim'] = df['window'].str.split('-', expand=True).astype(int).mean(axis=1)
    long_df = df.melt(id_vars=['center_dim'], value_vars=required_cols, var_name='split', value_name='score')
    long_df['split'] = long_df['split'].str.replace('_score', '')

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=long_df, x='center_dim', y='score', hue='split', marker='o')
    plt.title("Performance Trend Comparison Across Splits", fontsize=18)
    plt.savefig("split_trend_comparison_plot.png")
    plt.show()

def plot_multi_seed_analysis(df: pd.DataFrame, model_name: str, task_name: str):
    """Visualizes the results of the multi-seed region growing search."""
    if df.empty: return
    df["start_dim"] = df["window"].apply(lambda x: int(x.split("-")[0]))
    df = df.sort_values("start_dim")
    palette = {"Initial Scan": "lightblue", "Search Seed": "orange", "Grown Hotspot": "darkgreen", "Best Overall Hotspot": "crimson"}
    
    plt.figure(figsize=(16, 8))
    sns.barplot(data=df, x="window", y="score", hue="type", palette=palette, dodge=False)
    plt.title(f"Multi-Seed Region Growing Search\nModel: {model_name} | Task: {task_name}", fontsize=16)
    plt.savefig("multi_seed_region_growing_analysis.png")
    plt.show()

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sentence embedding performance.")
    parser.add_argument('--model_name', type=str, default="tomaarsen/mpnet-base-nli", help='The SentenceTransformer model to use.')
    parser.add_argument('--task_name', type=str, default="STSBenchmark", help='The MTEB task to evaluate on.')
    parser.add_argument('--analysis_mode', type=str, default="region_growing", choices=["sliding_window", "multi_split_sliding_window", "region_growing"], help='The analysis mode to run.')
    parser.add_argument('--window_size', type=int, default=64, help='Window size for sliding window analysis.')
    parser.add_argument('--step_size', type=int, default=16, help='Step size for sliding window analysis.')
    parser.add_argument('--initial_step', type=int, default=16, help='Initial step for region growing search.')
    parser.add_argument('--growth_increment', type=int, default=8, help='Growth increment for region growing search.')
    parser.add_argument('--num_seeds', type=int, default=3, help='Number of seeds for region growing search.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for encoding.')
    parser.add_argument('--max_workers', type=int, default=None, help='Max workers for parallel processing.')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}\nProcessing Model '{args.model_name}' for Task '{args.task_name}'\n{'='*70}")
    model = SentenceTransformer(args.model_name, device=device)
    task = mteb.get_tasks(tasks=[args.task_name])[0]
    task.load_data()
    metric = get_task_metric(task)
    total_dims = model.get_sentence_embedding_dimension()

    print("\nStep 1: Pre-computing embeddings...")
    all_precomputed_embeddings = create_precomputed_embeddings(task, model, args.batch_size)
    
    if args.analysis_mode == "sliding_window":
        params = {"window_size": args.window_size, "step_size": args.step_size}
        results_df = run_sliding_window_search(args.task_name, metric, all_precomputed_embeddings, params, total_dims, args.max_workers)
        plot_sliding_window_summary(results_df, args.model_name, args.task_name)
        analyze_and_visualize_distribution(results_df.copy())
    elif args.analysis_mode == "multi_split_sliding_window":
        params = {"window_size": args.window_size, "step_size": args.step_size}
        results_df = run_multi_split_sliding_window_search(args.task_name, metric, all_precomputed_embeddings, params, total_dims, args.max_workers)
        analyze_and_plot_split_comparison(results_df)
    elif args.analysis_mode == "region_growing":
        params = {"initial_step": args.initial_step, "growth_increment": args.growth_increment, "num_seeds": args.num_seeds}
        results_df = run_multi_seed_region_growing_search(args.task_name, metric, all_precomputed_embeddings, params, total_dims, args.max_workers)
        plot_multi_seed_analysis(results_df, args.model_name, args.task_name)
    else:
        print(f"ERROR: Unknown analysis_mode: {args.analysis_mode}")

    print("\nAll analyses complete.")