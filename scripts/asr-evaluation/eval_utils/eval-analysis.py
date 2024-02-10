import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_system_performance(data, metric='WER', save_path='system_performance.png'):
    """
    Plots a box plot showing the distribution of a specified metric per system across all datasets.

    Parameters:
    - data: pandas DataFrame containing the ASR evaluation results.
    - metric: The metric to use for the box plot (e.g., 'SER', 'WER').
    - save_path: The path where the plot image will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='system', y=metric, data=data)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Box Plot of {metric} Per System Across All Datasets')
    plt.xlabel('System')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_dataset_performance(data, metric='WER', save_path='dataset_performance.png'):
    """
    Plots a box plot showing the distribution of a specified metric per dataset across all systems.

    Parameters:
    - data: pandas DataFrame containing the ASR evaluation results.
    - metric: The metric to use for the box plot (e.g., 'SER', 'WER').
    - save_path: The path where the plot image will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dataset', y=metric, data=data)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Box Plot of {metric} Per Dataset Across All Systems')
    plt.xlabel('Dataset')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(eval_results, metric='WER', save_path='.'):
    # Load the data
    data = pd.read_csv(eval_results, sep='\t')
    today = pd.Timestamp.now().strftime('%Y%m%d')
    os.makedirs(os.path.join(save_path, today), exist_ok=True)

    # Plot and save the figures
    plot_system_performance(data, metric, os.path.join(save_path, today,'system_performance.png'))
    plot_dataset_performance(data, metric, os.path.join(save_path, today,'dataset_performance.png'))

def parse_arguments():
    parser = argparse.ArgumentParser(description='ASR Evaluation Analysis')
    parser.add_argument('--eval_results', type=str, help='Path to the TSV file')
    parser.add_argument('--metric', type=str, default='WER', help='Metric to plot (default: WER)')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save the plots')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.eval_results, args.metric, args.save_path)

# Example usage:
# python eval-analysis.py --file_path asr_eval_results.tsv --metric WER