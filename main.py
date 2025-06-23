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
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# --- 1. ADVANCED CONFIGURATION ---

# Define the models, tasks, and search parameters to test.
SEARCH_CONFIG = {
    "models": ["tomaarsen/mpnet-base-nli"],
    "tasks": ["STSBenchmark"],
    "sliding_window_params": [
        {"window_size": 64, "step_size": 32},
        {"window_size": 128, "step_size": 64},
    ],
    # --- Configuration for 'region_growing' search ---
    "region_growing_params": {
        "initial_step": 16,  # Coarse-grained scan step size
        "growth_increment": 8,  # How many dims to add when growing
        "num_seeds": 3,  # NEW: Number of top regions to grow from
    },
}

ENCODE_BATCH_SIZE = 128
MAX_WORKERS = None  # Use all available CPU cores. Set to a number to limit.

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

def create_precomputed_embeddings(
    task: mteb.AbsTask, model: SentenceTransformer
) -> Dict[str, Any]:
    """
    Analyzes the task type and creates the appropriate pre-computed embeddings.
    For pair tasks, it now also stores the original sentences.
    """
    print(f"INFO: Analyzing task '{task.metadata_dict["name"]}' of type {task.__class__.__name__}")
    precomputed_embeddings = {}
    
    is_pair_task = isinstance(task, (AbsTaskSTS, AbsTaskPairClassification))

    for split in task.metadata.eval_splits:
        dataset = task.dataset[split]
        
        if is_pair_task:
            print(f"INFO: Handling as a Pair Task (e.g., STS, PairClassification).")
            sentences1 = dataset["sentence1"]
            sentences2 = dataset["sentence2"]
            corpus = list(set(sentences1 + sentences2))
            print(f"INFO: Found {len(corpus)} unique sentences in '{split}' split for encoding.")
            
            encoded_corpus = model.encode(
                corpus, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
            )
            
            sent_to_emb = {sent: emb for sent, emb in zip(corpus, encoded_corpus)}
            
            embeddings1 = np.array([sent_to_emb[s] for s in sentences1])
            embeddings2 = np.array([sent_to_emb[s] for s in sentences2])
            
            # --- MODIFICATION HERE ---
            # Store everything needed to reconstruct the smart wrapper later
            precomputed_embeddings[split] = {
                "sents1": sentences1,
                "embs1": embeddings1,
                "sents2": sentences2,
                "embs2": embeddings2,
            }

        else: # Handles Classification, Clustering, etc.
            # This part remains the same and works correctly.
            print(f"INFO: Handling as a single-sentence list task (e.g., Classification, Clustering).")
            if 'sentences' in dataset.features:
                corpus = dataset['sentences']
            elif 'text' in dataset.features:
                corpus = dataset['text']
            elif 'query' in dataset.features:
                corpus = dataset['query']
            else:
                raise ValueError(f"Could not find a standard sentence column in task {task.metadata_dict["name"]}")

            print(f"INFO: Found {len(corpus)} sentences in '{split}' split for encoding.")

            embeddings = model.encode(
                corpus, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
            )
            precomputed_embeddings[split] = embeddings
            
    return precomputed_embeddings

def evaluate_single_slice_parallel(
    task_name: str,
    metric: str,
    all_embeddings: Dict[str, Any],
    dims: Tuple[int, int],
) -> Tuple[int, int, float]:
    """
    Evaluates a single slice using pre-computed embeddings. This version uses
    the appropriate wrapper for each task type to be fully compliant with the
    MTEB API.
    """
    start_dim, end_dim = dims
    task = mteb.get_tasks(tasks=[task_name])[0]
    eval_split = task.metadata.eval_splits[0]
    split_data = all_embeddings[eval_split]
    
    evaluator = MTEB(tasks=[task])
    
    # --- FINAL FIX START ---
    is_pair_structure = isinstance(split_data, dict) and 'sents1' in split_data

    if is_pair_structure:
        # This is a pair task. Create and use the SmartPairWrapper.
        sliced_embs1 = split_data['embs1'][:, start_dim:end_dim]
        sliced_embs2 = split_data['embs2'][:, start_dim:end_dim]

        # Build the map from each sentence to its sliced embedding
        sentence_map = {}
        for s, e in zip(split_data['sents1'], sliced_embs1):
            sentence_map[s] = e
        for s, e in zip(split_data['sents2'], sliced_embs2):
            sentence_map[s] = e
        
        smart_wrapper = SmartPairWrapper(sentence_map)
        
        # Pass the wrapper as the model, WITHOUT corpus_preprocessed=True
        results = evaluator.run(smart_wrapper, output_folder=None, eval_splits=[eval_split])
    else:
        # This is a single-sentence task. Use the original wrapper.
        # `split_data` is the NumPy array of embeddings.
        slice_wrapper = PrecomputedEmbeddingWrapper(split_data, start_dim, end_dim)
        results = evaluator.run(slice_wrapper, output_folder=None, eval_splits=[eval_split])
    # --- FINAL FIX END ---
    
    try:
        score = results[0].scores[eval_split][0][metric] * 100
        return start_dim, end_dim, score
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error extracting score in worker: {e}")
        return start_dim, end_dim, 0.0

def get_task_metric(task_object: Any) -> str:
    """
    Determines the appropriate MTEB metric based on the task's class type.
    This is the modern, robust way to handle different MTEB tasks.
    """
    if isinstance(task_object, AbsTaskSTS):
        # For Semantic Textual Similarity tasks
        return "spearman"
    elif isinstance(task_object, AbsTaskClassification):
        # For single-label Classification tasks
        return "accuracy"
    elif isinstance(task_object, AbsTaskClustering):
        # For Clustering tasks
        return "v_measure"
    elif isinstance(task_object, AbsTaskReranking):
        # For Reranking tasks
        return "map"
    elif isinstance(task_object, AbsTaskPairClassification):
        # For tasks like NLI that classify pairs of sentences
        # Often uses accuracy, but check the specific task if needed
        return "accuracy"
    else:
        # A sensible default if the task type is new or unhandled
        print(f"Warning: Unhandled task type: {task_object.__class__.__name__}. Falling back to 'main_score'.")
        return "main_score"


# --- 3. ANALYSIS ALGORITHMS (ADAPTED FOR PARALLELISM & MULTI-SEED) ---


# NEW: Worker function to grow a SINGLE seed. Runs in its own process.
def grow_single_seed_worker(
    task_name: str,
    metric: str,
    all_embeddings: Dict,
    seed_dims: Tuple[int, int],
    seed_score: float,
    growth_increment: int,
    total_dims: int,
) -> Dict:
    """Takes a single seed and performs the iterative growth algorithm."""
    current_start, current_end = seed_dims
    current_best_score = seed_score

    # Helper for fast, sequential evaluation within this worker
    def fast_evaluate(dims):
        _, _, score = evaluate_single_slice_parallel(
            task_name, metric, all_embeddings, dims
        )
        return score

    while True:
        grew_successfully = False
        # Try growing to the right
        right_dims = (current_start, min(current_end + growth_increment, total_dims))
        if right_dims[1] > current_end:
            right_score = fast_evaluate(right_dims)
            if right_score > current_best_score:
                current_end, current_best_score = right_dims[1], right_score
                grew_successfully = True

        # Try growing to the left
        left_dims = (max(current_start - growth_increment, 0), current_end)
        if left_dims[0] < current_start:
            left_score = fast_evaluate(left_dims)
            if left_score > current_best_score:
                current_start, current_best_score = left_dims[0], left_score
                grew_successfully = True

        if not grew_successfully:
            break  # Growth stopped, local maximum found for this seed

    return {
        "seed_window": f"{seed_dims[0]}-{seed_dims[1]}",
        "final_window": f"{current_start}-{current_end}",
        "final_score": current_best_score,
    }

# --- ADD THIS NEW CLASS ---
class SmartPairWrapper:
    """
    A wrapper that mimics a SentenceTransformer's .encode() method.
    It uses a pre-computed mapping of sentences to their embeddings to handle
    the separate sentence lists required by pair tasks (e.g., STS).
    It is stateless and safe for multiprocessing.
    """
    def __init__(self, sentence_map: Dict[str, np.ndarray]):
        self.sentence_map = sentence_map

    def encode(self, sentences: List[str], *args, **kwargs) -> np.ndarray:
        # Look up each sentence in the provided list and return its embedding
        return np.array([self.sentence_map[s] for s in sentences])

def run_multi_seed_region_growing_search(
    task_name: str, metric: str, all_embeddings: Dict, params: Dict, total_dims: int
) -> pd.DataFrame:
    """
    Performs a robust, multi-seed region growing search in parallel.
    """
    print("\n--- Running Smart Search: Multi-Seed Region Growing ---")
    initial_step = params["initial_step"]
    growth_increment = params["growth_increment"]
    num_seeds = params["num_seeds"]

    # 1. Coarse-grained initial scan (Parallel)
    print(f"1. Parallel coarse scan with step size: {initial_step}")
    initial_scan_results = []
    dims_to_evaluate = [
        (s, min(s + initial_step, total_dims))
        for s in range(0, total_dims, initial_step)
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                evaluate_single_slice_parallel,
                task_name,
                metric,
                all_embeddings,
                dims
            )
            for dims in dims_to_evaluate
        ]
        
        # --- FIX START: Collect the results from the completed futures ---
        for future in as_completed(futures):
            try:
                start_dim, end_dim, score = future.result()
                initial_scan_results.append({
                    "window": f"{start_dim}-{end_dim}",
                    "score": score
                })
            except Exception as exc:
                print(f"A worker process generated an exception: {exc}")
        # --- FIX END ---

    # Handle case where all workers might have failed
    if not initial_scan_results:
        print("ERROR: Initial scan did not produce any results. Cannot proceed.")
        return pd.DataFrame()

    scan_df = (
        pd.DataFrame(initial_scan_results)
        .sort_values(by="score", ascending=False)
        .reset_index(drop=True)
    )

    # 2. Select top N seeds
    top_seeds_df = scan_df.head(num_seeds)
    print(f"\n2. Selected top {len(top_seeds_df)} seeds for parallel growing:")
    for _, row in top_seeds_df.iterrows():
        print(f"  - Seed: {row['window']}, Initial Score: {row['score']:.2f}")

    # 3. Grow all seeds in parallel
    print("\n3. Growing all seeds in parallel...")
    final_hotspots = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        growth_futures = []
        for _, seed_row in top_seeds_df.iterrows():
            seed_dims = tuple(map(int, seed_row["window"].split("-")))
            future = executor.submit(
                grow_single_seed_worker,
                task_name,
                metric,
                all_embeddings,
                seed_dims,
                seed_row["score"],
                growth_increment,
                total_dims,
            )
            growth_futures.append(future)

        for future in as_completed(growth_futures):
            result = future.result()
            print(
                f"  - Growth complete. Seed {result['seed_window']} -> Final Hotspot {result['final_window']} (Score: {result['final_score']:.2f})"
            )
            final_hotspots.append(result)

    # 4. Prepare DataFrame for plotting
    plot_df = scan_df.copy()
    plot_df["type"] = "Initial Scan"

    hotspot_data = []
    for hotspot in final_hotspots:
        plot_df.loc[plot_df["window"] == hotspot["seed_window"], "type"] = "Search Seed"
        hotspot_data.append(
            {
                "window": hotspot["final_window"],
                "score": hotspot["final_score"],
                "type": "Grown Hotspot",
            }
        )

    plot_df = pd.concat([plot_df, pd.DataFrame(hotspot_data)], ignore_index=True)

    if not plot_df[plot_df["type"] == "Grown Hotspot"].empty:
        best_hotspot_idx = plot_df[plot_df["type"] == "Grown Hotspot"]["score"].idxmax()
        plot_df.loc[best_hotspot_idx, "type"] = "Best Overall Hotspot"

    return plot_df.sort_values(by="score", ascending=False)


# --- 4. VISUALIZATION FUNCTIONS (UPDATED FOR MULTI-SEED) ---


def plot_multi_seed_analysis(df: pd.DataFrame, model_name: str, task_name: str):
    """Visualizes the results of the multi-seed region growing search."""
    if df.empty:
        return
    print("\n--- Generating Multi-Seed Region Growing Analysis Plot ---")

    # Ensure window is treated as a categorical variable in plotting order
    df["start_dim"] = df["window"].apply(lambda x: int(x.split("-")[0]))
    df = df.sort_values("start_dim").reset_index(drop=True)

    # Define a clear color palette
    palette = {
        "Initial Scan": "lightblue",
        "Search Seed": "orange",
        "Grown Hotspot": "darkgreen",
        "Best Overall Hotspot": "crimson",
    }

    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=df, x="window", y="score", hue="type", palette=palette, dodge=False
    )

    plt.title(
        f"Multi-Seed Region Growing Search\nModel: {model_name} | Task: {task_name}",
        fontsize=16,
        weight="bold",
    )
    plt.xlabel("Dimension Window", fontsize=12)
    plt.ylabel("Performance Score (Spearman)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Search Stage", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("multi_seed_region_growing_analysis.png")
    plt.show()


# (The sliding window and its plot function remain unchanged)
def run_sliding_window_search(*args, **kwargs):
    # This function is defined in the previous answer and can be reused here.
    # For brevity, it is omitted, but it would be identical.
    pass


def plot_window_search_summary(*args, **kwargs):
    # This function is defined in the previous answer and can be reused here.
    pass


# --- 5. MAIN EXECUTION (REVISED) ---

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Add more tasks here to test robustness!
    # For example: SEARCH_CONFIG["tasks"] = ["STSBenchmark", "EmotionClassification"]
    
    for model_name in SEARCH_CONFIG["models"]:
        for task_name in SEARCH_CONFIG["tasks"]:
            main_start_time = time.time()
            print(
                f"\n{'=' * 70}\nProcessing Model '{model_name}' for Task '{task_name}'\n{'=' * 70}"
            )

            model = SentenceTransformer(model_name, device=device)
            task = mteb.get_tasks(tasks=[task_name])[0]
            metric = get_task_metric(task)
            total_dims = model.get_sentence_embedding_dimension()

            task.load_data()

            print("\nStep 1: Pre-computing embeddings (robustly) for the dataset...")
            # Use the new, robust function
            all_precomputed_embeddings = create_precomputed_embeddings(task, model)
            
            # --- The rest of your script remains the same ---
            # --- Run the Multi-Seed Region Growing "Smart" Search ---
            rg_params = SEARCH_CONFIG["region_growing_params"]
            multi_seed_df = run_multi_seed_region_growing_search(
                task.metadata_dict["name"],
                metric,
                all_precomputed_embeddings,
                rg_params,
                total_dims,
            )
            plot_multi_seed_analysis(multi_seed_df, model_name, task_name)

            del model, all_precomputed_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            main_end_time = time.time()
            print(
                f"\nTotal time for {model_name}/{task_name}: {main_end_time - main_start_time:.2f} seconds."
            )

    print("\nAll analyses complete.")