from pathlib import Path
import pickle
from tqdm import tqdm
from datetime import datetime
import torch
from linmult import LinMulT
from personalitylinmult.computational_complexity.utils import (
    load_mult, load_linmult, model_forward, load_batch
)
from personalitylinmult.computational_complexity.performance_metrics import (
    speedup_ratio, average_measurements, 
    measure_inference_flops_and_flops, measure_training_flops_and_flops
)
from exordium.utils.loss import bell_l2_l1_loss


def train_flops_experiment(output_dir):
    torch.set_float32_matmul_precision('medium') 
    device = torch.device('cuda:0')
    mult = load_mult().to(device)
    linmult = load_linmult().to(device)

    # Measure Inference FLOPs and FLOPS
    def inference_mult_func():
        return measure_inference_flops_and_flops(mult, batch)

    def inference_linmult_func():
        return measure_inference_flops_and_flops(linmult, batch)

    # Measure Training FLOPs and FLOPS
    def training_mult_func():
        return measure_training_flops_and_flops(mult,
                                                torch.optim.Adam(mult.parameters(), lr=0.001), 
                                                bell_l2_l1_loss, 
                                                batch, 
                                                torch.rand((1,5), device=device), 
                                                model_forward)

    def training_linmult_func():
        return measure_training_flops_and_flops(linmult,
                                                torch.optim.Adam(linmult.parameters(), lr=0.001),
                                                bell_l2_l1_loss,
                                                batch,
                                                torch.rand((1,5), device=device),
                                                model_forward)

    output = {}
    for tm in [0.5, 1.0, 1.5, 2.0]:
        batch = load_batch(batch_size=1, time_multiplier=tm)
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

    output_path = Path(output_dir) / 'training_flops_gpu.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    train_flops_experiment(output_dir='results/computational_complexity')