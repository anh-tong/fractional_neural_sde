import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_setting(font_size=10):

    """Set up matplotlib environment"""

    plt.rc("figure", dpi=256)
    plt.rc("font", family="serif", size=font_size)
    plt.rc("text", usetex=True)
    # plt.rc('text.latex', preamble=r'''
    #     \usepackage{amsmath,amsfonts}
    #     \newcommand{\m}[1]{\mathbf{#1}}
    #     \renewcommand{\v}[1]{\boldsymbol{#1}}''')


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


def _stable_division(a, b, epsilon=1e-5):
    b = torch.where(
        b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign()
    )
    return a / b


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
