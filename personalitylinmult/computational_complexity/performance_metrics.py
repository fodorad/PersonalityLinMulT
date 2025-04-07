import time
import numpy as np
from tqdm import tqdm
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
        latency = (time.time() - start_time) * 1000 # Convert to ms
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


def measure_inference_flops(model, batch):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        with_flops=True
    ) as prof:
        with torch.no_grad():
            model(batch)
    flops = prof.key_averages().total_average().flops
    return flops


def measure_training_flops(model, optimizer, loss_fn, batch, target, model_forward):
    model.train()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        with_flops=True
    ) as prof:
        optimizer.zero_grad()
        output = model_forward(model, batch)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    flops = prof.key_averages().total_average().flops
    return flops


def measure_inference_flops_and_flops(model, batch):
    # Measure FLOPs
    inference_flops = measure_inference_flops(model, batch)
    
    # Measure time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        model(batch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    inference_time = end_time - start_time # seconds
    flops_per_sec = inference_flops / inference_time if inference_time > 0 else float('inf')
    
    return inference_flops, inference_time, flops_per_sec


def measure_training_flops_and_flops(model, optimizer, loss_fn, batch, target, model_forward):
    model.train()
    
    # Measure FLOPs
    training_flops = measure_training_flops(model, optimizer, loss_fn, batch, target, model_forward)
    
    # Measure time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    optimizer.zero_grad()
    output = model_forward(model, batch)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    training_time = end_time - start_time # seconds
    flops_per_sec = training_flops / training_time if training_time > 0 else float('inf')
    
    return training_flops, training_time, flops_per_sec


def average_measurements(func, num_runs=10):
    flops_list = []
    flops_per_sec_list = []
    
    for _ in range(10):
        func()
    
    for _ in tqdm(range(num_runs)):
        flops, exec_time, flops_per_sec = func()
        flops_list.append(flops)
        flops_per_sec_list.append(flops_per_sec)
    
    flops_mean = np.mean(flops_list)
    flops_per_sec_mean = np.mean(flops_per_sec_list)
    return flops_mean, flops_per_sec_mean