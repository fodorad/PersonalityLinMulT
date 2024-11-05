import pickle
from tqdm import tqdm
import torch
from performance_metrics import measure_inference_time, percentage_reduction, speedup_ratio
from torch.utils.data import Dataset, DataLoader
from utils import load_mult, load_linmult, load_sample


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


def find_max_feasible_batch_size(model, device: str):
    torch.cuda.empty_cache()
    model = model.to(device)
    model.eval()

    for time_multiplier in [0.5, 1.0, 1.5]:
        print('Time_multiplier:', time_multiplier)
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            batch = load_sample(batch_size=batch_size, time_multiplier=time_multiplier)
            input_data = [torch.FloatTensor(inp).to(device) for inp in batch]

            try:
                with torch.no_grad():
                    output = model(input_data)
                print('Batch size:', input_data[0].shape[0])
            except:
                print('Batch size:', input_data[0].shape[0], 'CPU OOM!')
            
            del input_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision('highest') # this was the only option at the time of publication 
    device1 = torch.device('cpu')
    device2 = torch.device('cpu')

    mult = load_mult()
    linmult = load_linmult()

    features = 'oowfr' # oowfr
    mult_results_dict = {}
    linmult_results_dict = {}
    
    for time_multiplier in [0.5, 1.0, 1.5, 2.0]:
        print(f'{"-"*20}\nTime multiplier ({time_multiplier})')
        mult_results_dict[time_multiplier] = []
        linmult_results_dict[time_multiplier] = []

        for batch_size in [1, 2, 4]:
            print(f'{"-"*20}\nBatch ({batch_size})')

            batch = load_sample(features=features, batch_size=batch_size, time_multiplier=time_multiplier)

            try:
                mult_results, linmult_results = run_comparative_experiment(mult, linmult, batch, device1, device2)
                mult_results_dict[time_multiplier].append(mult_results)
                linmult_results_dict[time_multiplier].append(linmult_results)
                del batch

                with open(f'data/db_processed/fi/computational_complexity/inference_metrics_cpu_{features}.pkl', 'wb') as f:
                    pickle.dump({'MulT': mult_results_dict, 'LinMulT': linmult_results_dict}, f)
            except:
                print('CPU fail!')

        '''
        for batch_size in [16, 32, 64, 128, 256, 512, 1024]:
            print(f'{"-"*20}\nBatch ({batch_size})')
            batch = load_sample(batch_size=batch_size, time_multiplier=time_multiplier)

            try:
                linmult_results = run_inference_experiment(linmult, batch, device2)
                print(f"[LinMulT] Inference Time: {linmult_results['inference_time_mean']:.2f} \u00B1 {linmult_results['inference_time_std']:.2f} ms")
                print(f"[LinMulT] Memory Usage - Allocated: {linmult_results['peak_memory_allocated']:.2f} MB")
                print(f"[LinMulT] Memory Usage - Reserved: {linmult_results['peak_memory_reserved']:.2f} MB")
                print(f"[LinMulT] FLOPs: {linmult_results['flops']:.2e}")
                linmult_results_list.append(linmult_results)
                del batch

                with open(f'data/db_processed/fi/computational_complexity/inference_metrics_cpu_{str(time_multiplier)}.pkl', 'wb') as f:
                    pickle.dump({'MulT': mult_results_list, 'LinMulT': linmult_results_list}, f)
            except:
                print('CPU fail!')
        '''
    