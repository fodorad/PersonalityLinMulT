import yaml
import torch
import numpy as np
from linmult import LinMulT
from personalitylinmult import MODEL_DIR


def SentimentLinMulT() -> tuple[LinMulT, dict]:
    config = load_yaml_config('config/MOSEI/model_OOWFR_LinMulT.yaml')
    model = LinMulT(config=config)
    weight_path = MODEL_DIR / 'sentiment' / 'checkpoint_sentiment.ckpt'
    weights = torch.load(weight_path, weights_only=True, map_location=torch.device('cpu'))['state_dict']
    weights = {k.replace("model.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    standardization_params = load_standardization_params()
    return model, standardization_params


def load_standardization_params():
    d = {}
    data = np.load(str(MODEL_DIR / 'sentiment' / 'standardization' / 'egemaps.npz'))
    d['egemaps_lld'] = {}
    d['egemaps_lld']['mean'] = torch.FloatTensor(data['mean'])
    d['egemaps_lld']['std'] = torch.FloatTensor(data['std'])
    data = np.load(str(MODEL_DIR / 'sentiment' / 'standardization' / 'opengraphau.npz'))
    d['opengraphau'] = {}
    d['opengraphau']['mean'] = torch.FloatTensor(data['mean'])
    d['opengraphau']['std'] = torch.FloatTensor(data['std'])
    return d


def load_yaml_config(file_path):
    """Loads a YAML configuration file."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{file_path}': {e}")
