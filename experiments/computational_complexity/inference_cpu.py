import pickle
from tqdm import tqdm
import torch
from personalitylinmult.computational_complexity.performance_metrics import (
    measure_inference_time, percentage_reduction, speedup_ratio
)
from personalitylinmult.computational_complexity.utils import (
    load_mult, load_linmult, load_batch
)


def run_inference_experiment(model, batch, device: str):
    torch.cuda.empty_cache()
    model = model.to(device)
    model.eval()
    batch = [torch.FloatTensor(inp).to(device) for inp in batch]

    # Measure inference time
    inference_time_mean, inference_time_std = measure_inference_time(model, batch)

    del batch

    return {
        'inference_time_mean': inference_time_mean,
        'inference_time_std': inference_time_std
    }


def run_comparative_experiment(mult_model, linmult_model, batch, device1, device2):

    # Run experiments for LinMulT
    print(f'LinMulT run_inference_experiment on {device2}')
    linmult_results = run_inference_experiment(linmult_model, batch, device2)

    # Run experiments for MulT
    print(f'MulT run_inference_experiment on {device1}')
    mult_results = run_inference_experiment(mult_model, batch, device1)

    # Calculate metrics
    speedup = speedup_ratio(mult_results['inference_time_mean'], linmult_results['inference_time_mean'])
    time_reduction = percentage_reduction(mult_results['inference_time_mean'], linmult_results['inference_time_mean'])
    
    # Display results
    print(f"[MulT]    Inference Time: {mult_results['inference_time_mean']:.2f} \u00B1 {mult_results['inference_time_std']:.2f} ms")
    print(f"[LinMulT] Inference Time: {linmult_results['inference_time_mean']:.2f} \u00B1 {linmult_results['inference_time_std']:.2f} ms")
    print(f"[LinMulT] Speedup Ratio: {speedup:.2f}x")
    print(f"[LinMulT] Time Reduction: {time_reduction:.2f}%")

    return mult_results, linmult_results


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cpu')

    mult = load_mult()
    linmult = load_linmult()

    mult_results_dict = {}
    linmult_results_dict = {}
    
    for time_multiplier in [0.5, 1.0, 1.5, 2.0]:
        print(f'{"-"*20}\nTime multiplier ({time_multiplier})')
        mult_results_dict[time_multiplier] = []
        linmult_results_dict[time_multiplier] = []

        for batch_size in [1, 2, 4, 8]:
            print(f'{"-"*20}\nBatch ({batch_size})')
            batch = load_batch(batch_size=batch_size, time_multiplier=time_multiplier)

            mult_results, linmult_results = run_comparative_experiment(mult, linmult, batch, device, device)
            mult_results_dict[time_multiplier].append(mult_results)
            linmult_results_dict[time_multiplier].append(linmult_results)
            del batch

            with open(f'results/computational_complexity/inference_metrics_cpu.pkl', 'wb') as f:
                pickle.dump({'MulT': mult_results_dict, 'LinMulT': linmult_results_dict}, f)
