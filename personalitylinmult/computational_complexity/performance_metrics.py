import time
import numpy as np
import torch
import torch.profiler


def measure_inference_time(model, batch):
    # Warm-up to stabilize performance
    for _ in range(10):
        with torch.no_grad():
            _ = model(batch)

    latency_list = []
    for _ in range(10):
        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            _ = model(batch)

        torch.cuda.synchronize()
        latency = (time.time() - start_time) * 1000  # Convert to ms
        latency_list.append(latency)

    latency_array = np.array(latency_list)
    return np.mean(latency_array), np.std(latency_array)


def percentage_reduction(original, new):
    if original == 0: return 0
    return ((original - new) / original) * 100


def speedup_ratio(time1, time2):
    if time2 == 0: return 0
    return time1 / time2


def measure_memory_usage(model, batch, device: str):
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(batch)

    peak_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)  # Convert to MB

    return peak_memory_allocated, peak_memory_reserved


def measure_flops_with_profiler(model, batch):
    # Use torch.profiler to measure FLOPs
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        with_flops=True
    ) as profiler:
        with torch.no_grad():
            model(batch)

    # Print the profiler key metrics
    flops = profiler.key_averages().total_average().flops
    return flops


def _measure_training_flops_with_profiler(model, batch, target, loss_fn):
    # Enable profiling to measure both forward and backward FLOPs
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        with_flops=True
    ) as profiler:
        
        # Run the forward pass
        output = model(batch)
        
        # Calculate the loss
        loss = loss_fn(output, target)

        # Backward pass to calculate gradients
        loss.backward()

    # Get the total FLOPs for both forward and backward passes (training FLOPs)
    flops = profiler.key_averages().total_average().flops
    return flops