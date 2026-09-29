import pandas as pd
import matplotlib.pyplot as plt
from ..utils import get_artifact_dir
import logging

logger = logging.getLogger(__name__)

def compare_methods():
    results_dir = get_artifact_dir("results")
    
    methods = ["scratch", "residual", "safe"]
    dfs = []
    
    for method in methods:
        csv_path = results_dir / f"eval_{method}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['method'] = method
            dfs.append(df)
            
    if not dfs:
        logger.error("No evaluation results found to compare.")
        return
        
    all_results = pd.concat(dfs, ignore_index=True)
    
    summary = all_results.groupby("method").agg({
        "success": "mean",
        "collisions": lambda x: (x > 0).mean(),
        "reward": ["mean", "std"],
        "length": "mean",
        "min_clearance": "mean",
        "energy_proxy": "mean"
    }).reset_index()
    
    summary.columns = [
        "method", "success_rate", "collision_rate", "mean_reward", 
        "std_reward", "mean_episode_length", "mean_min_clearance", 
        "mean_energy_proxy"
    ]
    
    summary.to_csv(results_dir / "comparison.csv", index=False)
    logger.info("Comparison saved to artifacts/results/comparison.csv")
    
    plot_dir = get_artifact_dir("plots")
    
    metrics = {
        "success_rate": "Success Rate",
        "collision_rate": "Collision Rate",
        "mean_reward": "Mean Reward",
        "mean_energy_proxy": "Mean Energy Proxy",
        "mean_min_clearance": "Mean Min Clearance"
    }
    
    for col, title in metrics.items():
        plt.figure(figsize=(8, 5))
        methods_in_summary = summary["method"].tolist()
        colors = ['blue', 'orange', 'green'][:len(methods_in_summary)]
        plt.bar(summary["method"], summary[col], color=colors)
        plt.title(title)
        plt.ylabel(title)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}_comparison.png")
        plt.close()
