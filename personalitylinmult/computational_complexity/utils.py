import numpy as np
from linmult import LinMulT


def model_forward(model, batch):
    return model(batch)[0] # LinMulT specific


def load_mult():
    model_config = 'config/FI/model_OOWFR_MulT.yaml'
    model = LinMulT(model_config)
    return model


def load_linmult():
    model_config = 'config/FI/model_OOWFR_LinMulT.yaml'
    model = LinMulT(model_config)
    return model


def load_batch(batch_size: int, time_multiplier: float):

    # oowfr
    return [
        np.random.rand(batch_size, int(1500 * time_multiplier), 25),
        np.random.rand(batch_size, int(450 * time_multiplier), 41),
        np.random.rand(batch_size, int(1500 * time_multiplier), 768),
        np.random.rand(batch_size, int(450 * time_multiplier), 256),
        np.random.rand(batch_size, int(80 * time_multiplier), 1024),
    ]


if __name__ == "__main__":
    batch_normal = load_batch(batch_size=1, time_multiplier=1)
    batch_short = load_batch(batch_size=4, time_multiplier=0.5)
    batch_long = load_batch(batch_size=8, time_multiplier=1.5)
    print('normal batch shape:', [elem.shape for elem in batch_normal])
    print('short batch shape:', [elem.shape for elem in batch_short])
    print('long batch shape:', [elem.shape for elem in batch_long])