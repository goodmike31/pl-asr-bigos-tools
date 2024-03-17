import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_data(file_path):
    """Load TSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, sep='\t')

def plot_audio_duration_distribution(df, save_path):
    print("Plot the distribution of audio durations")
    plt.figure(figsize=(10, 6))
    plt.hist(df['audio_duration'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Audio Durations')
    plt.xlabel('Audio Duration (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    file_path = os.path.join(save_path, 'audio_duration_distribution.png')
    print("Save path: ", file_path)

    plt.savefig(file_path)

def plot_average_wer_by_duration_bucket(df, bucket_size, save_path):
    print("Plot the average WER for audio duration buckets.")
    # Create a new column for duration buckets
    df['duration_bucket'] = (df['audio_duration'] // bucket_size) * bucket_size
    # Group by the new bucket column and calculate the mean WER
    avg_wer_by_bucket = df.groupby('duration_bucket')['WER'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(avg_wer_by_bucket['duration_bucket'], avg_wer_by_bucket['WER'], width=bucket_size, color='lightgreen', edgecolor='black')
    plt.title(f'Average WER by Audio Duration Buckets (Size: {bucket_size} seconds)')
    plt.xlabel('Audio Duration Bucket (seconds)')
    plt.ylabel('Average WER')
    plt.grid(True)
    plt.show()
    file_path = os.path.join(save_path, 'average_wer_by_duration_bucket.png')
    print("Save path: ", file_path)

    plt.savefig(file_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='ASR Evaluation Analysis')
    parser.add_argument('--input_file', type=str, help='Path to the TSV file')
    parser.add_argument('--bucket_size', type=float, help='Bucket size in seconds')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save the plots')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Load data
    df = load_data(args.input_file)

    print(df.head())
    # Plot distribution of audio durations
    plot_audio_duration_distribution(df, args.save_path)
    
    # Plot average WER by duration bucket
    plot_average_wer_by_duration_bucket(df, args.bucket_size, args.save_path)
