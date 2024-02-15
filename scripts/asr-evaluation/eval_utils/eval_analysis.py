import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import os

def boxplot_performance(data, dim, metric, save_path):
    """
    Plots a box plot showing the distribution of a specified metric per provided dimension e.g. dataset, system, split etc.

    Parameters:
    - data: pandas DataFrame containing the ASR evaluation results.
    - dimn: The dimension to use for the box plot (e.g., 'dataset', 'system').
    - metric: The metric to use for the box plot (e.g., 'SER', 'WER').
    - save_path: The path where the plot image will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dim, y=metric, data=data)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Box Plot of {metric} Per {dim}')
    plt.xlabel(dim)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def boxplot_wer_per_dataset(eval_results, save_path='./data/eval_plots'):
    # Load the data
    data = pd.read_csv(eval_results, sep='\t')
    today = pd.Timestamp.now().strftime('%Y%m%d')
    os.makedirs(os.path.join(save_path, today), exist_ok=True)

    # Plot and save the figures
    boxplot_performance(data, "dataset", "WER", os.path.join(save_path, "WER-across-systems.png")

def parse_arguments():
    parser = argparse.ArgumentParser(description='ASR Evaluation Analysis')
    parser.add_argument('--analysis_type', type=str, help='WER_PER_AUDIO_DURATION', "WER_PER_SYSTEM", "WER_PER_DATASET")
    parser.add_argument('--eval_results', type=str, help='Path to the TSV file')
    parser.add_argument('--dim', type=str, default='dataset', help='Dimension to plot (default: dataset)')
    parser.add_argument('--metric', type=str, default='WER', help='Metric to plot (default: WER)')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save the plots')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main_agg(args.eval_results, args.dim, args.metric, args.save_path)

# python eval-analysis.py --file_path asr_eval_results.tsv --metric WER

