"""Adapt from
https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
"""
import math

import torch
import torch.nn as nn
from fractional_neural_sde.fractional_noise import SparseGPNoise
from fractional_neural_sde.utils import _stable_division
from torch import distributions
from torchsde import BaseSDE


class LatentSDE(BaseSDE):
    def __init__(
        self,
        noise_path: SparseGPNoise,
        theta: float = 1.0,
        mu: float = 1.0,
        sigma: float = 0.5,
        noise_type="diagonal",
        sde_type="ito",
    ):
        super().__init__(noise_type, sde_type)

        self.noise_path = noise_path
        # prior SDE
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # mean and var of initial point
        log_var = math.log(sigma**2 / (2 * theta))
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[log_var]]))

        # posterior drift
        self.net = nn.Sequential(
            nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1)
        )
        self.net[-1].weight.data.fill_(0.0)
        self.net[-1].bias.data.fill_(0.0)

        # posterior of initial point
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[log_var]]), requires_grad=True)

    def precompute_white_noise(self, batch_size):
        """Precompute Cholesky for white noise"""
        self.noise_path.precompute(batch_size)

    def f_and_g(self, t, y):
        """Drift and diffusion"""
        mean, sqrt_var_dt = self.noise_path(t)

        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)

        x = torch.cat([torch.sin(t), torch.cos(t), y], dim=-1)
        f = self.net(x)
        g = self.sigma.repeat(y.size(0), 1)

        # new drift and diffusion in Eq. 12
        f = f + g * mean
        g = g * sqrt_var_dt
        return f, g

    def h(self, t, y):
        r"""Drift of prior SDE

        dX_t = h(t, X_t)dt + \sigma(t, X_t)dB_t
        """
        return self.theta * (self.mu - y)

    def f_and_g_aug(self, t, y):
        """Augment drift and diffusion to compute KL while solving SDE

        The output of drift function `f` is added a quantity from Giranov's theorem
        The output of diffusion `g` is added a zero value
        """
        y = y[:, 0:1]

        f, g = self.f_and_g(t, y)
        h = self.h(t, y)

        # g can be very small sometime so that the following is more numerically stable
        u = _stable_division(f - h, g, 1e-3)

        # augmented drift
        f_logqp = 0.5 * (u**2).sum(dim=1, keepdim=True)
        f_aug = torch.cat([f, f_logqp], dim=1)

        # augmented diffusion
        g_logqp = torch.zeros_like(y)
        g_aug = torch.cat([g, g_logqp], dim=1)
        return f_aug, g_aug

    def sample_q(self, ts, batch_size, sdeint_fn, method, dt, bm, eps: None):
        """Sample posterior"""

        if eps is None:
            eps = torch.randn(batch_size, 1).to(self.qy0_mean)

        # sample initial point
        y0 = self.qy0_mean + eps * self.qy0_std
        # make sure precompute inducing before solving SDE
        self.precompute_white_noise(batch_size=batch_size)
        # return the solution of solver with posterior drift and diffusion
        return sdeint_fn(
            sde=self,
            y0=y0,
            ts=ts,
            bm=bm,
            method=method,
            dt=dt,
            names={"drift_and_diffusion": "f_and_g"},
        )

    @property
    def py0_std(self):
        return torch.exp(0.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(0.5 * self.qy0_logvar)

    def forward(self, ts, batch_size, sdeint_fn, method, dt, bm=None, eps=None):
        """SDE integration

        Args:
            ts: time step at which solution will return
            batch_size: batch size
            sdeint_fn: `torchsde` SDE solver. Normally, we use `euler`
            dt: step size of SDE solver
            bm: Brownian motion
            eps: noise for intial point
        """
        if eps is None:
            eps = torch.randn(batch_size, 1).to(self.qy0_std)

        # sample initial point and compute KL at t=0
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)

        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)

        # compute cholesky and inducing sample before solving
        self.precompute_white_noise(batch_size=batch_size)
        # solve the path from 0 -> T
        aug_ys = sdeint_fn(
            sde=self,
            bm=bm,
            y0=aug_y0,
            ts=ts,
            method=method,
            dt=dt,
            rtol=1e-3,
            atol=1e-3,
            names={"drift_and_diffusion": "f_and_g_aug"},
        )

        # seperate ys and log pq
        ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        logqp = (logqp0 + logqp_path).mean(dim=0)
        return ys, logqp
