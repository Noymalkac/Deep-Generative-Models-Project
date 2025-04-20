
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
# from scipy.stats import ttest_ind
from scipy.stats import ttest_rel  # for paired t-test
import matplotlib.image as mpimg


def load_metrics(file_path):
    df = pd.read_csv(file_path)
    df = df[df["CD_Loss"].notna() & df["MSE_Bias_Loss"].notna() & df["MSE_Var_Loss"].notna()]
    df = df.reset_index(drop=True)
    return df


def summarize_metrics(df):
    return {
        "final_cd_loss": df["CD_Loss"].iloc[-1],
        "final_bias_mse": df["MSE_Bias_Loss"].iloc[-1],
        "final_var_mse": df["MSE_Var_Loss"].iloc[-1],
        "avg_cd_loss": df["CD_Loss"].mean(),
        "avg_bias_mse": df["MSE_Bias_Loss"].mean(),
        "avg_var_mse": df["MSE_Var_Loss"].mean(),
    }


def compare_final_metrics(original_metrics, fast_metrics):
    print("\n\033[1m Final Metrics Comparison\033[0m")
    headers = ["Version", "Final CD Loss", "Final Bias MSE", "Final Var MSE"]
    print("{:<10} {:<15} {:<15} {:<15}".format(*headers))
    for label, metrics in zip(["Original", "Altered"], [original_metrics, fast_metrics]):
        print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            label,
            metrics["final_cd_loss"],
            metrics["final_bias_mse"],
            metrics["final_var_mse"],
        ))


def compare_average_metrics(original_metrics, fast_metrics):
    print("\n\033[1m Average Metrics Comparison\033[0m")
    headers = ["Version", "Avg CD Loss", "Avg Bias MSE", "Avg Var MSE"]
    print("{:<10} {:<15} {:<15} {:<15}".format(*headers))
    for label, metrics in zip(["Original", "Altered"], [original_metrics, fast_metrics]):
        print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            label,
            metrics["avg_cd_loss"],
            metrics["avg_bias_mse"],
            metrics["avg_var_mse"],
        ))


def load_epoch_times(file_path):
    return pd.read_csv(file_path)


def summarize_times(df):
    return {
        "total_time_sec": df["time_sec"].sum(),
        "avg_time_sec": df["time_sec"].mean(),
        "num_epochs": len(df)
    }


def compare_times(original_times, fast_times):
    print("\n Training Time Summary")
    headers = ["Version", "Total Time (sec)", "Avg Time/Epoch (sec)", "Epochs"]
    print("{:<10} {:<20} {:<25} {:<10}".format(*headers))
    for label, times in zip(["Original", "Altered"], [original_times, fast_times]):
        print("{:<10} {:<20.2f} {:<25.2f} {:<10}".format(
            label, times["total_time_sec"], times["avg_time_sec"], times["num_epochs"]
        ))
    total_speedup = (1 - fast_times["total_time_sec"] / original_times["total_time_sec"]) * 100
    print("\n Speed-Up Summary:")
    print(f"Total training time reduced by: \033[1m{total_speedup:.2f}%\033[0m")


def plot_metrics(df_orig, df_fast, out_dir="comparison_results"):
    os.makedirs(out_dir, exist_ok=True)

    # Plot: CD Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df_orig["epoch"], df_orig["CD_Loss"], label="Original - CD Loss")
    plt.plot(df_fast["epoch"], df_fast["CD_Loss"], label="Altered - CD Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CD Loss")
    plt.title("CD Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cd_loss_comparison.png")
    plt.close()

    # Plot: Bias and Variance MSE
    plt.figure(figsize=(10, 6))
    plt.plot(df_orig["epoch"], df_orig["MSE_Bias_Loss"], label="Original - Bias MSE")
    plt.plot(df_fast["epoch"], df_fast["MSE_Bias_Loss"], label="Altered - Bias MSE")
    plt.plot(df_orig["epoch"], df_orig["MSE_Var_Loss"], label="Original - Var MSE", linestyle="--")
    plt.plot(df_fast["epoch"], df_fast["MSE_Var_Loss"], label="Altered - Var MSE", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mse_loss_comparison.png")
    plt.close()


def plot_histograms(df_orig, df_fast, out_dir="comparison_results"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot: CD Loss
    plt.figure(figsize=(10, 6))
    plt.hist(df_orig["CD_Loss"], bins=30, alpha=0.6, label="Original CD")
    plt.hist(df_fast["CD_Loss"], bins=30, alpha=0.6, label="Altered CD")
    plt.title(f"Histogram of CD_Loss")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/CD_histogram.png")
    plt.close()

    # Plot: Bias and Variance MSE
    plt.figure(figsize=(10, 6))
    plt.hist(df_orig["MSE_Bias_Loss"], bins=30, alpha=0.6, label="Original_MSE_Bias")
    plt.hist(df_fast["MSE_Bias_Loss"], bins=30, alpha=0.6, label="Altered_MSE_Bias")
    plt.hist(df_orig["MSE_Var_Loss"], bins=30, alpha=0.6, label="Original_MSE_Var")
    plt.hist(df_fast["MSE_Var_Loss"], bins=30, alpha=0.6, label="Altered_MSE_Var")
    plt.title(f"Histogram of MSE_Bias_Loss and MSE_Var_Loss")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/MSE_histogram.png")
    plt.close()

def statistical_tests(df_orig, df_fast):
    print("\n Statistical Significance Tests (Paired t-test)")
    print("{:<20} {:<15} {}".format("Metric", "p-value", "Result"))

    for col in ["CD_Loss", "MSE_Bias_Loss", "MSE_Var_Loss"]:
        vals_orig = df_orig[col].reset_index(drop=True)
        vals_fast = df_fast[col].reset_index(drop=True)
        
        min_len = min(len(vals_orig), len(vals_fast))
        vals_orig = vals_orig[:min_len]
        vals_fast = vals_fast[:min_len]

        if len(vals_orig) > 1 and len(vals_fast) > 1:
            stat, pval = ttest_rel(vals_orig, vals_fast)
            result = "Significant" if pval < 0.05 else "Not significant"
            print(f"{col:<20} {pval:<15.4e} {result}")
        else:
            print(f"{col:<20} Insufficient data")



def plot_weight_comparison(original_img_path, fast_img_path, out_dir="comparison_results"):
    os.makedirs(out_dir, exist_ok=True)

    img_orig = mpimg.imread(original_img_path)
    img_fast = mpimg.imread(fast_img_path)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_orig)
    plt.title("Original Model - Weights (Last Epoch)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_fast)
    plt.title("Altered Model - Weights (Last Epoch)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{out_dir}/weights_comparison_epoch_49.png")
    plt.close()
    print("Saved: weights_comparison_epoch_49.png")




if __name__ == "__main__":

    orig_metrics_path = r"C:\Users\noyco\Downloads\neural-boltzmann-machines-main\.experiment_results\logs\lightning_logs\version_0\metrics.csv"
    fast_metrics_path = r"C:\Users\noyco\Downloads\neural-boltzmann-machines-main\.experiment_results\logs\lightning_logs\version_1\metrics.csv"
    orig_times_path = r"C:\Users\noyco\Downloads\neural-boltzmann-machines-main\.experiment_results\logs\lightning_logs\version_0\epoch_times.csv"
    fast_times_path = r"C:\Users\noyco\Downloads\neural-boltzmann-machines-main\.experiment_results\logs\lightning_logs\version_1\epoch_times.csv"

    df_original = load_metrics(orig_metrics_path)
    df_fast = load_metrics(fast_metrics_path)

    original_summary = summarize_metrics(df_original)
    fast_summary = summarize_metrics(df_fast)

    df_times_original = load_epoch_times(orig_times_path)
    df_times_fast = load_epoch_times(fast_times_path)

    time_summary_original = summarize_times(df_times_original)
    time_summary_fast = summarize_times(df_times_fast)

    compare_final_metrics(original_summary, fast_summary)
    compare_average_metrics(original_summary, fast_summary)
    compare_times(time_summary_original, time_summary_fast)

    plot_weight_comparison(
    original_img_path= r"C:\Users\noyco\Downloads\neural-boltzmann-machines-main\.experiment_results\logs\lightning_logs\version_0\weights\weights_epoch_49.png",
    fast_img_path= r"C:\Users\noyco\Downloads\neural-boltzmann-machines-main\.experiment_results\logs\lightning_logs\version_1\weights\weights_epoch_49.png")


    plot_metrics(df_original, df_fast)
    plot_histograms(df_original, df_fast)
    statistical_tests(df_original, df_fast)
    print("\nComparison complete. Results saved to 'comparison_results/' folder.")
