import torch
import numpy as np
from linmult import LinMulT
from personalitylinmult import MODEL_DIR
from personalitylinmult.train.parser import load_yaml_config


def PersonalityLinMulT() -> tuple[LinMulT, dict]:
    config = load_yaml_config('config/FI/model_OOWFR_LinMulT.yaml')
    model = LinMulT(config=config)
    weight_path = MODEL_DIR / 'app' / 'checkpoint_app.ckpt'
    weights = torch.load(weight_path, weights_only=True, map_location=torch.device('cpu'))['state_dict']
    weights = {k.replace("model.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    standardization_params = load_standardization_params()
    return model, standardization_params


def load_standardization_params():
    d = {}
    data = np.load(str(MODEL_DIR / 'app' / 'standardization' / 'egemaps_lld.npz'))
    d['egemaps_lld'] = {}
    d['egemaps_lld']['mean'] = torch.FloatTensor(data['mean'])
    d['egemaps_lld']['std'] = torch.FloatTensor(data['std'])
    data = np.load(str(MODEL_DIR / 'app' / 'standardization' / 'opengraphau.npz'))
    d['opengraphau'] = {}
    d['opengraphau']['mean'] = torch.FloatTensor(data['mean'])
    d['opengraphau']['std'] = torch.FloatTensor(data['std'])
    return d
