import time
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch
from torch.profiler import profile, ProfilerActivity
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from linmult import LinMulT
from performance_metrics import measure_memory_usage
from personalitylinmult.train.train import ModelWrapper, load_config
from personalitylinmult.train.fi import OOWFRDataModule
from utils import load_mult, load_linmult, load_sample
from performance_metrics import speedup_ratio
from exordium.utils.loss import bell_l2_l1_loss


def measure_inference_flops(model, batch):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        with torch.no_grad():
            model(batch)
    flops = prof.key_averages().total_average().flops
    return flops

def measure_training_flops(model, batch, target):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = bell_l2_l1_loss
    model.train()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        optimizer.zero_grad()
        output = model(batch)[0] # mult and linmult specific
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
    
    inference_time = end_time - start_time  # In seconds
    flops_per_sec = inference_flops / inference_time if inference_time > 0 else float('inf')
    
    return inference_flops, inference_time, flops_per_sec

def measure_training_flops_and_flops(model, batch, target):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = bell_l2_l1_loss
    model.train()
    
    # Measure FLOPs
    training_flops = measure_training_flops(model, batch, target)
    
    # Measure time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    optimizer.zero_grad()
    output = model(batch)[0] # mult and linmult specific
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    training_time = end_time - start_time  # In seconds
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

def train_flops_experiment():
    torch.set_float32_matmul_precision('highest') 
    device = torch.device('cuda:0')
    mult = load_mult().to(device)
    linmult = load_linmult().to(device)
    target = torch.rand((1,5), device=device)

    # Measure Inference FLOPs and FLOPS
    def inference_mult_func():
        return measure_inference_flops_and_flops(mult, batch)

    def inference_linmult_func():
        return measure_inference_flops_and_flops(linmult, batch)

    # Measure Training FLOPs and FLOPS
    def training_mult_func():
        return measure_training_flops_and_flops(mult, batch, target)

    def training_linmult_func():
        return measure_training_flops_and_flops(linmult, batch, target)

    output = {}
    for tm in [0.5, 1.0, 1.5, 2.0]:
        batch = load_sample(features='oowfr', batch_size=1, time_multiplier=tm)
        batch = [torch.FloatTensor(elem).to(device) for elem in batch]

        mult_inference_flops_mean, \
            mult_inference_flops_per_sec_mean = average_measurements(
                inference_mult_func, num_runs=10)
        print(f"MulT inference FLOPs mean: {mult_inference_flops_mean:.2e}")
        print(f"MulT inference FLOPS mean: {mult_inference_flops_per_sec_mean:.2e}")
        linmult_inference_flops_mean, \
            linmult_inference_flops_per_sec_mean = average_measurements(
                inference_linmult_func, num_runs=10)
        print(f"LinMulT inference FLOPs mean: {linmult_inference_flops_mean:.2e}")
        print(f"LinMulT inference FLOPS mean: {linmult_inference_flops_per_sec_mean:.2e}")

        speedup_inference_flops = speedup_ratio(mult_inference_flops_mean, linmult_inference_flops_mean)
        speedup_inference_flops_per_sec = speedup_ratio(mult_inference_flops_per_sec_mean, linmult_inference_flops_per_sec_mean)
        print('speedup inference FLOPs:', speedup_inference_flops)
        print('speedup inference FLOPS:', speedup_inference_flops_per_sec)

        mult_training_flops_mean, \
            mult_training_flops_per_sec_mean = average_measurements(
                training_mult_func, num_runs=10)
        print(f"MulT training FLOPs mean: {mult_training_flops_mean:.2e}")
        print(f"MulT training FLOPS mean: {mult_training_flops_per_sec_mean:.2e}")
        linmult_training_flops_mean, \
            linmult_training_flops_per_sec_mean = average_measurements(
                training_linmult_func, num_runs=10)
        print(f"LinMulT training FLOPs mean: {linmult_training_flops_mean:.2e}")
        print(f"LinMulT training FLOPS mean: {linmult_training_flops_per_sec_mean:.2e}")

        output[tm] = {}
        output[tm]['MulT'] = {
            'inference_flops_mean': mult_inference_flops_mean,
            'inference_flops_per_sec_mean': mult_inference_flops_per_sec_mean,
            'training_flops_mean': mult_training_flops_mean,
            'training_flops_per_sec_mean': mult_training_flops_per_sec_mean,
        }
        output[tm]['LinMulT'] = {
            'inference_flops_mean': linmult_inference_flops_mean,
            'inference_flops_per_sec_mean': linmult_inference_flops_per_sec_mean,
            'training_flops_mean': linmult_training_flops_mean,
            'training_flops_per_sec_mean': linmult_training_flops_per_sec_mean,
        }

    with open('data/db_processed/fi/computational_complexity/training_flops_gpu_oowfr.pkl', 'wb') as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    train_flops_experiment()