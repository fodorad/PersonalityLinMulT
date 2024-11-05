from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from performance_metrics import speedup_ratio, percentage_reduction

custom_blue = (0.855, 0.91, 0.988)
custom_green = (0.835, 0.91, 0.831)
line_width = 2
border_blue = (0.424, 0.557, 0.749)
border_green = (0.51, 0.702, 0.40)


def plot_inference_complexity(mult_results, linmult_results):
    # Example data (replace with your actual data)
    batch_sizes_mult = [1, 2, 3, 4, 5, 6, 7, 8]
    batch_sizes_linmult = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]

    # Inference time (ms) - replace with actual mean and std lists
    mult_inference_time_mean = [elem['inference_time_mean'] for elem in mult_results]
    mult_inference_time_std = [elem['inference_time_std'] for elem in mult_results]
    linmult_inference_time_mean = [elem['inference_time_mean'] for elem in linmult_results]
    linmult_inference_time_std = [elem['inference_time_std'] for elem in linmult_results]

    # FLOPs - replace with actual FLOPs values
    mult_flops = [elem['flops'] for elem in mult_results]
    linmult_flops = [elem['flops'] for elem in linmult_results]

    # Memory usage (MB) - replace with actual memory usage values
    mult_memory_allocated_usage = [elem['peak_memory_allocated'] for elem in mult_results]
    linmult_memory_allocated_usage = [elem['peak_memory_allocated'] for elem in linmult_results]
    mult_memory_reserved_usage = [elem['peak_memory_reserved'] for elem in mult_results]
    linmult_memory_reserved_usage = [elem['peak_memory_reserved'] for elem in linmult_results]

    # Speedup ratio and memory reduction - replace with actual calculated values
    speedup = [speedup_ratio(mult_elem['inference_time_mean'], linmult_elem['inference_time_mean']) for mult_elem, linmult_elem in zip(mult_results, linmult_results)]
    memory_reduction_allocated = [percentage_reduction(mult_elem['peak_memory_allocated'], linmult_elem['peak_memory_allocated']) for mult_elem, linmult_elem in zip(mult_results, linmult_results)]
    memory_reduction_reserved = [percentage_reduction(mult_elem['peak_memory_reserved'], linmult_elem['peak_memory_reserved']) for mult_elem, linmult_elem in zip(mult_results, linmult_results)]

    # Creating the figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid of subplots
    fig.tight_layout(pad=3.0)  # Adjust padding between subplots

    # Plot 1: Inference time comparison
    axs[0, 0].errorbar(batch_sizes_mult, mult_inference_time_mean, yerr=mult_inference_time_std,
                    fmt='-o', color='blue', label='MulT', capsize=5)
    axs[0, 0].errorbar(batch_sizes_linmult, linmult_inference_time_mean, yerr=linmult_inference_time_std,
                    fmt='-o', color='green', label='LinMulT', capsize=5)
    axs[0, 0].set_title('Inference Time Comparison')
    axs[0, 0].set_xlabel('Batch Size')
    axs[0, 0].set_ylabel('Time (ms)')
    axs[0, 0].set_xticks(batch_sizes_linmult)
    axs[0, 0].set_xticklabels(batch_sizes_linmult)
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Plot 2: FLOPs comparison
    axs[0, 1].plot(batch_sizes_mult, mult_flops, '-o', color='blue', label='MulT')
    axs[0, 1].plot(batch_sizes_linmult, linmult_flops, '-o', color='green', label='LinMulT')
    axs[0, 1].set_title('FLOPs Comparison')
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('FLOPs')
    axs[0, 1].set_xticks(batch_sizes_linmult)
    axs[0, 1].set_xticklabels(batch_sizes_linmult)
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plot 3: Memory usage comparison
    axs[1, 0].plot(batch_sizes_mult, mult_memory_allocated_usage, '-o', color='blue', label='MulT')
    axs[1, 0].plot(batch_sizes_linmult, linmult_memory_allocated_usage, '-o', color='green', label='LinMulT')
    axs[1, 0].plot(batch_sizes_mult, mult_memory_reserved_usage, ':o', color='blue', label='MulT')
    axs[1, 0].plot(batch_sizes_linmult, linmult_memory_reserved_usage, ':o', color='green', label='LinMulT')
    axs[1, 0].set_title('Memory Usage Comparison')
    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Memory Usage (MB)')
    axs[1, 0].set_xticks(batch_sizes_linmult)
    axs[1, 0].set_xticklabels(batch_sizes_linmult)
    axs[1, 0].legend()
    axs[1, 0].grid()

    # Plot 4: Speedup ratio and memory reduction comparison
    ax4 = axs[1, 1]
    ax4.plot(batch_sizes_mult, speedup, '-o', color='blue', label='Speedup Ratio')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Speedup Ratio', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.set_xticks(batch_sizes_linmult)
    ax4.set_xticklabels(batch_sizes_linmult)
    ax4.grid()

    # Create a twin axis for memory reduction
    ax4b = ax4.twinx()
    ax4b.plot(batch_sizes_mult, memory_reduction_allocated, '-s', color='green', label='Memory allocated Reduction (%)')
    ax4b.plot(batch_sizes_mult, memory_reduction_reserved, ':s', color='green', label='Memory reserved Reduction (%)')
    ax4b.set_ylabel('Memory Reduction (%)', color='green')
    ax4b.set_ylim([0, 100])
    ax4b.set_yticks(list(range(0, 101, 10)))
    ax4b.tick_params(axis='y', labelcolor='green')

    # Title and legends for the comparison plot
    ax4.set_title('Speedup Ratio and Memory Reduction Comparison')
    ax4.legend(loc='upper left')
    ax4b.legend(loc='lower right')

    # Adjust layout to remove excessive free space
    plt.tight_layout()

    # Save the figure using Path from pathlib
    output_path = Path("output") / "scalability_plots.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(output_path, bbox_inches='tight')  # Save figure with tight bounding box
    plt.close()


def plot_train_complexity():
    # Example data: Replace these with the actual data from the callbacks
    batch_sizes = [8, 64]  # Maximum feasible batch sizes for MulT and LinMulT
    training_times = [callback_mult.train_epoch_times, callback_linmult.train_epoch_times]
    validation_times = [callback_mult.validation_epoch_times, callback_linmult.validation_epoch_times]
    peak_memory = [max(callback_mult.peak_memory_usage), max(callback_linmult.peak_memory_usage)]
    flops = [sum(callback_mult.flops_per_epoch), sum(callback_linmult.flops_per_epoch)]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Time per Epoch
    axs[0, 0].bar(['MulT', 'LinMulT'], [np.mean(training_times[0]), np.mean(training_times[1])], color=['blue', 'green'])
    axs[0, 0].set_title('Average Training Time per Epoch')
    axs[0, 0].set_ylabel('Time (seconds)')

    # Plot 2: Total Training Time
    axs[0, 1].bar(['MulT', 'LinMulT'], [sum(training_times[0]) + sum(validation_times[0]), sum(training_times[1]) + sum(validation_times[1])], color=['blue', 'green'])
    axs[0, 1].set_title('Total Training Time')
    axs[0, 1].set_ylabel('Time (seconds)')

    # Plot 3: Peak Memory Usage
    axs[1, 0].bar(['MulT', 'LinMulT'], peak_memory, color=['blue', 'green'])
    axs[1, 0].set_title('Peak Memory Usage')
    axs[1, 0].set_ylabel('Memory (MB)')

    # Plot 4: FLOPs Required for Training
    axs[1, 1].bar(['MulT', 'LinMulT'], flops, color=['blue', 'green'])
    axs[1, 1].set_title('FLOPs Required for Training')
    axs[1, 1].set_ylabel('FLOPs')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig('train_complexity.png')
    plt.close()


def plot_inference_batch():
    print('\nCreating plots about inference time across batch sizes...')

    with open('data/db_processed/fi/computational_complexity/inference_metrics_gpu_oowfr.pkl', 'rb') as f:
        gpu_data = pickle.load(f)

    with open('data/db_processed/fi/computational_complexity/inference_metrics_cpu_oowfr.pkl', 'rb') as f:
        cpu_data = pickle.load(f)

    batch_sizes = [1, 2, 4, 8]
    time_multiplier = 1.0

    # Prepare the data for plotting
    gpu_mult_mean = [item['inference_time_mean'] for item in gpu_data['MulT'][time_multiplier]]
    gpu_mult_std = [item['inference_time_std'] for item in gpu_data['MulT'][time_multiplier]]
    gpu_linmult_mean = [item['inference_time_mean'] for item in gpu_data['LinMulT'][time_multiplier]]
    gpu_linmult_std = [item['inference_time_std'] for item in gpu_data['LinMulT'][time_multiplier]]

    print('GPU MulT inference time mean for batch size 1, 2, 4, 8:', gpu_mult_mean)
    print('GPU MulT inference time std for batch size 1, 2, 4, 8:', gpu_mult_std)
    print('GPU LinMulT inference time std for batch size 1, 2, 4, 8:', gpu_linmult_mean)
    print('GPU LinMulT inference time std for batch size 1, 2, 4, 8:', gpu_linmult_std)

    cpu_mult_mean = [item['inference_time_mean'] for item in cpu_data['MulT'][time_multiplier]]
    cpu_mult_std = [item['inference_time_std'] for item in cpu_data['MulT'][time_multiplier]]
    cpu_linmult_mean = [item['inference_time_mean'] for item in cpu_data['LinMulT'][time_multiplier]]
    cpu_linmult_std = [item['inference_time_std'] for item in cpu_data['LinMulT'][time_multiplier]]

    print('CPU MulT inference time mean for batch size 1, 2, 4, 8:', cpu_mult_mean)
    print('CPU MulT inference time std for batch size 1, 2, 4, 8:', cpu_mult_std)
    print('CPU LinMulT inference time std for batch size 1, 2, 4, 8:', cpu_linmult_mean)
    print('CPU LinMulT inference time std for batch size 1, 2, 4, 8:', cpu_linmult_std)

    gpu_speedup = [speedup_ratio(gpu_mult_mean[i], gpu_linmult_mean[i]) for i in range(len(gpu_mult_mean))]
    cpu_speedup = [speedup_ratio(cpu_mult_mean[i], cpu_linmult_mean[i]) for i in range(len(cpu_mult_mean))]

    print('GPU speedup of LinMulT compared to MulT for batch size 1, 2, 4, 8:',gpu_speedup)
    print('CPU speedup of LinMulT compared to MulT for batch size 1, 2, 4, 8:',cpu_speedup)

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Inference time as a bar plot
    axis_label_size = 14
    title_size = 16
    bar_width = 0.30
    space_between_bars = 0.03
    indices = np.arange(len(batch_sizes))

    axs[0].bar(indices[:len(gpu_mult_mean)], gpu_mult_mean, bar_width, yerr=gpu_mult_std, label='MulT', color=custom_blue, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_blue, linewidth=line_width)
    axs[0].bar(indices[:len(gpu_linmult_mean)] + bar_width + space_between_bars, gpu_linmult_mean, bar_width, yerr=gpu_linmult_std, label='LinMulT', color=custom_green, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_green, linewidth=line_width)

    for i in range(len(gpu_mult_mean)):
        speedup_val = f"{gpu_speedup[i]:.2f}x"
        height = max(gpu_mult_mean[i], gpu_linmult_mean[i])
        axs[0].text(indices[i] + (bar_width + space_between_bars) / 2, height + 15, speedup_val, ha='center', fontsize=12)

    axs[0].set_xlabel('Batch size', fontsize=axis_label_size)
    axs[0].set_ylabel('Inference time (ms)', fontsize=axis_label_size)
    axs[0].set_ylim([0,400])
    axs[0].set_xticks(indices[:len(gpu_linmult_mean)] + (bar_width + space_between_bars)/2)
    axs[0].set_xticklabels(batch_sizes[:len(gpu_linmult_mean)])
    axs[0].set_title('GPU inference time with speedup across batch sizes', fontsize=title_size)
    axs[0].legend(loc='upper left', framealpha=1.0)
    axs[0].grid(True)

    # Plotting bar for CPU inference time
    axs[1].bar(indices[:len(cpu_mult_mean)], cpu_mult_mean, bar_width, yerr=cpu_mult_std, label='MulT', color=custom_blue, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_blue, linewidth=line_width)
    axs[1].bar(indices[:len(cpu_linmult_mean)] + bar_width + space_between_bars, cpu_linmult_mean, bar_width, yerr=cpu_linmult_std, label='LinMulT', color=custom_green, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_green, linewidth=line_width)

    for i in range(len(cpu_mult_mean)):
        speedup_val = f"{cpu_speedup[i]:.2f}x"
        height = max(cpu_mult_mean[i], cpu_linmult_mean[i])
        axs[1].text(indices[i] + (bar_width + space_between_bars) / 2, height + 150, speedup_val, ha='center', fontsize=12)

    axs[1].set_xlabel('Batch size', fontsize=axis_label_size)
    axs[1].set_ylabel('Inference time (ms)', fontsize=axis_label_size)
    axs[1].set_yticks(range(0, 7500, 500))
    axs[1].set_xticks(indices[:len(cpu_linmult_mean)] + (bar_width + space_between_bars)/2)
    axs[1].set_xticklabels(batch_sizes[:len(cpu_linmult_mean)])
    axs[1].set_title('CPU inference time with speedup across batch sizes', fontsize=title_size)
    axs[1].legend(loc='upper left', framealpha=1.0)
    axs[1].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout(pad=2.0)
    plt.savefig("output/plot_inference_batch.png")
    plt.close()


def plot_inference_window():
    print('\nCreating plots about inference time across window sizes...')

    with open('data/db_processed/fi/computational_complexity/inference_metrics_gpu_oowfr.pkl', 'rb') as f:
        gpu_data = pickle.load(f)

    with open('data/db_processed/fi/computational_complexity/inference_metrics_cpu_oowfr.pkl', 'rb') as f:
        cpu_data = pickle.load(f)

    window_sizes = [7.5, 15, 22.5, 30]
    ws = [0.5, 1.0, 1.5, 2.0]

    # Prepare the data for plotting
    gpu_mult_mean = [gpu_data['MulT'][w][0]['inference_time_mean'] for w in ws]
    gpu_mult_std = [gpu_data['MulT'][w][0]['inference_time_std'] for w in ws]
    gpu_linmult_mean = [gpu_data['LinMulT'][w][0]['inference_time_mean'] for w in ws]
    gpu_linmult_std = [gpu_data['LinMulT'][w][0]['inference_time_std'] for w in ws]

    print('GPU MulT inference time mean for window size 7.5s, 15s, 22.5s, 30s:', gpu_mult_mean)
    print('GPU MulT inference time std for window size 7.5s, 15s, 22.5s, 30s:', gpu_mult_std)
    print('GPU LinMulT inference time std for window size 7.5s, 15s, 22.5s, 30s:', gpu_linmult_mean)
    print('GPU LinMulT inference time std for window size 7.5s, 15s, 22.5s, 30s:', gpu_linmult_std)

    cpu_mult_mean = [cpu_data['MulT'][w][0]['inference_time_mean'] for w in ws]
    cpu_mult_std = [cpu_data['MulT'][w][0]['inference_time_std'] for w in ws]
    cpu_linmult_mean = [cpu_data['LinMulT'][w][0]['inference_time_mean'] for w in ws]
    cpu_linmult_std = [cpu_data['LinMulT'][w][0]['inference_time_std'] for w in ws]

    print('CPU MulT inference time mean for window size 7.5s, 15s, 22.5s, 30s:', cpu_mult_mean)
    print('CPU MulT inference time std for window size 7.5s, 15s, 22.5s, 30s:', cpu_mult_std)
    print('CPU LinMulT inference time std for window size 7.5s, 15s, 22.5s, 30s:', cpu_linmult_mean)
    print('CPU LinMulT inference time std for window size 7.5s, 15s, 22.5s, 30s:', cpu_linmult_std)

    gpu_speedup = [speedup_ratio(gpu_mult_mean[i], gpu_linmult_mean[i]) for i in range(len(gpu_mult_mean))]
    cpu_speedup = [speedup_ratio(cpu_mult_mean[i], cpu_linmult_mean[i]) for i in range(len(cpu_mult_mean))]

    print('GPU speedup of LinMulT compared to MulT for window size 7.5s, 15s, 22.5s, 30s:', gpu_speedup)
    print('CPU speedup of LinMulT compared to MulT for window size 7.5s, 15s, 22.5s, 30s:', cpu_speedup)

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Inference time as a bar plot
    axis_label_size = 14
    title_size = 16
    bar_width = 0.30
    space_between_bars = 0.03
    indices = np.arange(len(window_sizes))

    axs[0].bar(indices[:len(gpu_mult_mean)], gpu_mult_mean, bar_width, yerr=gpu_mult_std, label='MulT', color=custom_blue, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_blue, linewidth=line_width)
    axs[0].bar(indices[:len(gpu_linmult_mean)] + bar_width + space_between_bars, gpu_linmult_mean, bar_width, yerr=gpu_linmult_std, label='LinMulT', color=custom_green, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_green, linewidth=line_width)

    for i in range(len(gpu_mult_mean)):
        speedup_val = f"{gpu_speedup[i]:.2f}x"
        height = max(gpu_mult_mean[i], gpu_linmult_mean[i])
        axs[0].text(indices[i] + (bar_width + space_between_bars) / 2, height + 12, speedup_val, ha='center', fontsize=12)

    axs[0].set_xlabel('Window size (s)', fontsize=axis_label_size)
    axs[0].set_ylabel('Inference time (ms)', fontsize=axis_label_size)
    axs[0].set_ylim([0,200])
    axs[0].set_xticks(indices[:len(gpu_linmult_mean)] + (bar_width + space_between_bars)/2)
    axs[0].set_xticklabels(window_sizes[:len(gpu_linmult_mean)])
    axs[0].set_title('GPU inference time with speedup across batch sizes', fontsize=title_size)
    axs[0].legend(loc='upper left', framealpha=1.0)
    axs[0].grid(True)

    # Plotting bar for CPU inference time
    axs[1].bar(indices[:len(cpu_mult_mean)], cpu_mult_mean, bar_width, yerr=cpu_mult_std, label='MulT', color=custom_blue, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_blue, linewidth=line_width)
    axs[1].bar(indices[:len(cpu_linmult_mean)] + bar_width + space_between_bars, cpu_linmult_mean, bar_width, yerr=cpu_linmult_std, label='LinMulT', color=custom_green, alpha=1.0, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, edgecolor=border_green, linewidth=line_width)

    for i in range(len(cpu_mult_mean)):
        speedup_val = f"{cpu_speedup[i]:.2f}x"
        height = max(cpu_mult_mean[i], cpu_linmult_mean[i])
        axs[1].text(indices[i] + (bar_width + space_between_bars) / 2, height + 250, speedup_val, ha='center', fontsize=12)

    axs[1].set_xlabel('Window size (s)', fontsize=axis_label_size)
    axs[1].set_ylabel('Inference time (ms)', fontsize=axis_label_size)
    axs[1].set_yticks(range(0, 7000, 500))
    axs[1].set_xticks(indices[:len(cpu_linmult_mean)] + (bar_width + space_between_bars)/2)
    axs[1].set_xticklabels(window_sizes[:len(cpu_linmult_mean)])
    axs[1].set_title('CPU inference time with speedup across window sizes', fontsize=title_size)
    axs[1].legend(loc='upper left', framealpha=1.0)
    axs[1].grid(True)
    
    # Adjust layout and show the plot
    plt.tight_layout(pad=2.0)
    plt.savefig("output/plot_inference_window.png")
    plt.close()


def plot_memory_comparison():

    with open('data/db_processed/fi/computational_complexity/inference_metrics_gpu_oowfr.pkl', 'rb') as f:
        gpu_data = pickle.load(f)

    # Sequence sizes (time multipliers)
    batch_sizes = [1, 2, 4, 8]
    sequence_multipliers = [0.5, 1.0, 1.5, 2.0]
    window_sizes = [7.5, 15, 22.5, 30]

    print('batch sizes:', batch_sizes)
    print('window sizes:', window_sizes)

    # Prepare data for allocated and reserved memory - batch size
    batch_mult_allocated = [gpu_data['MulT'][1.0][i]['peak_memory_allocated'] for i in range(len(batch_sizes))]
    batch_linmult_allocated = [gpu_data['LinMulT'][1.0][i]['peak_memory_allocated'] for i in range(len(batch_sizes))]

    batch_mult_reserved = [gpu_data['MulT'][1.0][i]['peak_memory_reserved'] for i in range(len(batch_sizes))]
    batch_linmult_reserved = [gpu_data['LinMulT'][1.0][i]['peak_memory_reserved'] for i in range(len(batch_sizes))]

    print('left plot: window size is 15s, batch size is changing')
    print('mult peak memory allocated:', batch_mult_allocated)
    print('linmult peak memory allocated:', batch_linmult_allocated)
    print('mult peak memory reserved:', batch_mult_reserved)
    print('linmult peak memory reserved:', batch_linmult_reserved)

    # data for sequence length
    mult_allocated = [gpu_data['MulT'][tm][0]['peak_memory_allocated'] for tm in sequence_multipliers]
    linmult_allocated = [gpu_data['LinMulT'][tm][0]['peak_memory_allocated'] for tm in sequence_multipliers]

    mult_reserved = [gpu_data['MulT'][tm][0]['peak_memory_reserved'] for tm in sequence_multipliers]
    linmult_reserved = [gpu_data['LinMulT'][tm][0]['peak_memory_reserved'] for tm in sequence_multipliers]

    print('right plot: window size is changing, batch size is 1')
    print('mult peak memory allocated:', mult_allocated)
    print('linmult peak memory allocated:', linmult_allocated)
    print('mult peak memory reserved:', mult_reserved)
    print('linmult peak memory reserved:', linmult_reserved)

    # Calculate percentage reduction
    batch_allocated_reduction = [(1 - batch_linmult_allocated[i] / batch_mult_allocated[i]) * 100 for i in range(len(batch_sizes))]
    batch_reserved_reduction = [(1 - batch_linmult_reserved[i] / batch_mult_reserved[i]) * 100 for i in range(len(batch_sizes))]

    print('percentage reduction for left plot')
    print('allocated reduction:', batch_allocated_reduction)
    print('reserved reduction:', batch_reserved_reduction)

    allocated_reduction = [(1 - linmult_allocated[i] / mult_allocated[i]) * 100 for i in range(len(sequence_multipliers))]
    reserved_reduction = [(1 - linmult_reserved[i] / mult_reserved[i]) * 100 for i in range(len(sequence_multipliers))]

    print('percentage reduction for right plot')
    print('allocated reduction:', allocated_reduction)
    print('reserved reduction:', reserved_reduction)

    # Bar plot width and indices
    axis_label_size = 14
    title_size = 16
    bar_width = 0.30
    space_between_bars = 0.03

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))


    indices = np.arange(len(batch_sizes))

    # Overlay reserved memory with the same alpha
    axs[0].bar(indices, batch_mult_reserved, bar_width, label='MulT (Reserved)', color=custom_blue, alpha=1.0, hatch='//', edgecolor=border_blue, linewidth=line_width)
    axs[0].bar(indices + bar_width + space_between_bars, batch_linmult_reserved, bar_width, label='LinMulT (Reserved)', color=custom_green, alpha=1.0, hatch='//', edgecolor=border_green, linewidth=line_width)

    # Plot allocated memory
    axs[0].bar(indices, batch_mult_allocated, bar_width, label='MulT (Allocated)', color=custom_blue, alpha=1.0, edgecolor=border_blue, linewidth=line_width)
    axs[0].bar(indices + bar_width + space_between_bars, batch_linmult_allocated, bar_width, label='LinMulT (Allocated)', color=custom_green, alpha=1.0, edgecolor=border_green, linewidth=line_width)

    # Add percentage reduction text on top of bars
    for i in range(len(batch_mult_allocated)):
        height = max(batch_mult_reserved[i], batch_linmult_reserved[i])
        reduction_text = f"{batch_allocated_reduction[i]:.2f}% [{batch_reserved_reduction[i]:.2f}%]"
        axs[0].text(indices[i] + (bar_width + space_between_bars) / 2, height + 20, reduction_text, ha='center', fontsize=11)

    axs[0].set_xlabel('Batch size', fontsize=axis_label_size)
    axs[0].set_ylabel('GPU memory usage (MB)', fontsize=axis_label_size)
    axs[0].set_ylim([0, 1700])
    axs[0].set_yticks(range(0, 1700, 100))
    axs[0].set_xticks(indices + (bar_width + space_between_bars) / 2)
    axs[0].set_xticklabels(batch_sizes)
    axs[0].set_title('GPU memory usage across batch sizes', fontsize=title_size)
    axs[0].legend(loc='upper left', framealpha=1.0)
    axs[0].grid(True)


    indices = np.arange(len(sequence_multipliers))

    # Overlay reserved memory with the same alpha
    axs[1].bar(indices, mult_reserved, bar_width, label='MulT (Reserved)', color=custom_blue, alpha=1.0, hatch='//', edgecolor=border_blue, linewidth=line_width)
    axs[1].bar(indices + bar_width + space_between_bars, linmult_reserved, bar_width, label='LinMulT (Reserved)', color=custom_green, alpha=1.0, hatch='//', edgecolor=border_green, linewidth=line_width)

    # Plot allocated memory
    axs[1].bar(indices, mult_allocated, bar_width, label='MulT (Allocated)', color=custom_blue, alpha=1.0, edgecolor=border_blue, linewidth=line_width)
    axs[1].bar(indices + bar_width + space_between_bars, linmult_allocated, bar_width, label='LinMulT (Allocated)', color=custom_green, alpha=1.0, edgecolor=border_green, linewidth=line_width)

    # Add percentage reduction text on top of bars
    for i in range(len(mult_allocated)):
        height = max(mult_reserved[i], linmult_reserved[i])
        reduction_text = f"{allocated_reduction[i]:.2f}% [{reserved_reduction[i]:.2f}%]"
        axs[1].text(indices[i] + (bar_width + space_between_bars) / 2, height + 20, reduction_text, ha='center', fontsize=11)

    axs[1].set_xlabel('Window size (s)', fontsize=axis_label_size)
    axs[1].set_ylabel('GPU memory usage (MB)', fontsize=axis_label_size)
    axs[1].set_ylim([0, 900])
    axs[1].set_yticks(range(0, 900, 100))
    axs[1].set_xticks(indices + (bar_width + space_between_bars) / 2)
    axs[1].set_xticklabels(window_sizes)
    axs[1].set_title('GPU memory usage across window sizes', fontsize=title_size)
    axs[1].legend(loc='upper left', framealpha=1.0)
    axs[1].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout(pad=2.0)
    plt.savefig("output/plot_inference_memory.png")
    plt.close()


def plot_flops_window():
    with open('data/db_processed/fi/computational_complexity/training_flops_gpu_oowfr.pkl', 'rb') as f:
        flops_data = pickle.load(f)

    # Sequence sizes (time multipliers) and batch sizes
    sequence_multipliers = [0.5, 1.0, 1.5, 2.0]
    window_sizes = [7.5, 15, 22.5, 30]

    print('window sizes:', window_sizes)

    # Prepare data for inference and training FLOPs
    mult_inference_flops = [flops_data[tm]['MulT']['inference_flops_mean'] for tm in sequence_multipliers]
    linmult_inference_flops = [flops_data[tm]['LinMulT']['inference_flops_mean'] for tm in sequence_multipliers]

    mult_training_flops = [flops_data[tm]['MulT']['training_flops_mean'] for tm in sequence_multipliers]
    linmult_training_flops = [flops_data[tm]['LinMulT']['training_flops_mean'] for tm in sequence_multipliers]

    print('FLOPs calculation for a single sample during inference across window durations')
    print('mult flops:', mult_inference_flops)
    print('linmult flops:', linmult_inference_flops)
    print('FLOPs calculation for a single sample during training across window durations')
    print('mult flops:', mult_training_flops)
    print('linmult flops:', linmult_training_flops)

    # Calculate percentage reduction in FLOPs between MulT and LinMulT
    inference_flops_reduction = [(1 - linmult_inference_flops[i] / mult_inference_flops[i]) * 100 for i in range(len(sequence_multipliers))]
    training_flops_reduction = [(1 - linmult_training_flops[i] / mult_training_flops[i]) * 100 for i in range(len(sequence_multipliers))]

    print('percentage reduction in FLOPs:')
    print('inference:', inference_flops_reduction)
    print('training:', training_flops_reduction)

    # Plot settings
    bar_width = 0.30
    space_between_bars = 0.03
    axis_label_size = 14
    title_size = 16
    line_width = 1.5


    fig, axs = plt.subplots(1, 1, figsize=(7, 5))


    indices = np.arange(len(sequence_multipliers))

    # Overlay reserved memory with the same alpha
    axs.bar(indices, mult_training_flops, bar_width, label='MulT (training)', color=custom_blue, alpha=1.0, hatch='//', edgecolor=border_blue, linewidth=line_width)
    axs.bar(indices + bar_width + space_between_bars, linmult_training_flops, bar_width, label='LinMulT (training)', color=custom_green, alpha=1.0, hatch='//', edgecolor=border_green, linewidth=line_width)

    # Plot allocated memory
    axs.bar(indices, mult_inference_flops, bar_width, label='MulT (inference)', color=custom_blue, alpha=1.0, edgecolor=border_blue, linewidth=line_width)
    axs.bar(indices + bar_width + space_between_bars, linmult_inference_flops, bar_width, label='LinMulT (inference)', color=custom_green, alpha=1.0, edgecolor=border_green, linewidth=line_width)

    # Add percentage reduction text on top of bars
    for i in range(len(mult_inference_flops)):
        height = max(mult_training_flops[i], linmult_training_flops[i])
        reduction_text = f"{inference_flops_reduction[i]:.2f}% [{training_flops_reduction[i]:.2f}%]"
        axs.text(indices[i] + (bar_width + space_between_bars) / 2, height + 0.05e+11, reduction_text, ha='center', fontsize=11)

    axs.set_xlabel('Window size (s)', fontsize=axis_label_size)
    axs.set_ylabel('FLOPs', fontsize=axis_label_size)
    axs.set_ylim([0, 2.5e+11])
    axs.set_yticks(np.arange(0, 2.5e+11 + 1, 0.25e+11))
    axs.set_xticks(indices + (bar_width + space_between_bars) / 2)
    axs.set_xticklabels(window_sizes)
    axs.set_title('FLOPs across window sizes', fontsize=title_size)
    axs.legend(loc='upper left', framealpha=1.0)
    axs.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout(pad=2.0)
    plt.savefig("output/plot_flops.png")
    plt.close()


if __name__ == "__main__":
    plot_flops_window()
    #plot_inference_batch()
    #plot_inference_window()
    #plot_memory_comparison()
