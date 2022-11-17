"""
Fractional Brownian motion in Score-Based Generative models

This file is adapted from
    https://github.com/google-research/torchsde/blob/master/examples/cont_ddpm.py
"""
import abc
import logging
import math
import os

import fire
import numpy as np
import torch
import torchdiffeq
import torchsde
import torchvision as tv
import tqdm
import wandb
from fractional_neural_sde import unet
from fractional_neural_sde.fractional_noise import SparseGPNoise
from torch import nn, optim
from torch.utils import data


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def fill_tail_dims(y: torch.Tensor, y_like: torch.Tensor):
    """Fill in missing trailing dimensions for y according to y_like."""
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]


class Module(abc.ABC, nn.Module):
    """A wrapper module that's more convenient to use."""

    def __init__(self):
        super(Module, self).__init__()
        self._checkpoint = False

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    @property
    def device(self):
        return next(self.parameters()).device


class ScoreMatchingSDE(Module):
    """Wraps score network with analytical sampling and cond. score computation.
    The variance preserving formulation in
        Score-Based Generative Modeling through Stochastic Differential Equations
        https://arxiv.org/abs/2011.13456
    """

    def __init__(
        self,
        denoiser,
        white_noise,
        input_size=(1, 28, 28),
        t0=0.0,
        t1=1.0,
        beta_min=0.1,
        beta_max=50.0,
    ):
        super(ScoreMatchingSDE, self).__init__()
        if t0 > t1:
            raise ValueError(f"Expected t0 <= t1, but found t0={t0:.4f}, t1={t1:.4f}")

        self.input_size = input_size
        self.denoiser = denoiser
        self.white_noise = white_noise

        self.t0 = t0
        self.t1 = t1

        self.beta_min = beta_min
        self.beta_max = beta_max

        # discretization to compute integral of \alpha = \beta * h^2
        self.register_buffer("steps", torch.linspace(self.t0, self.t1, 100))

    def precompute_white_noise(self, batch_size=1):
        """Precompute inducing points"""
        self.white_noise.precompute(batch_size)

    def init(self):
        """Precommpute Cholesky"""
        self.precompute_white_noise()
        var = torch.stack([self.white_noise(t)[1].pow(2) for t in self.steps], dim=0)
        #
        self._var = var.squeeze().clamp(1e-5)

    @property
    def white_noise_var(self):
        return self._var

    def original_f(self, t, y):
        return -0.5 * self._beta(t) * y

    def original_g(self, t, y):
        sqrt_beta = self._beta(t).sqrt()
        return fill_tail_dims(sqrt_beta, y).expand_as(y)

    def score(self, t, y):
        if isinstance(t, float):
            t = y.new_tensor(t)
        if t.dim() == 0:
            t = t.repeat(y.shape[0])
        return self.denoiser(t, y)

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _indefinite_int(self, t):
        r"""Indefinite integral.

        Note that the fractional case we commpute the integral

            \int \beta(t)\varsigma(t) dt
        """

        if not torch.is_tensor(t):
            t = self.steps.new_tensor([t])

        beta_steps = self._beta(self.steps).unsqueeze(0)  # size: 1, num_step
        diff = t.unsqueeze(-1) - self.steps.unsqueeze(0)
        indicator = (diff > 0).float()  # size: batch size, num step

        ret = indicator * beta_steps * self.white_noise_var.unsqueeze(0)
        ret = ret.sum(-1) * (self.steps[1] - self.steps[0])

        return ret

    def analytical_mean(self, t, x_t0):
        r"""Compute the analytic mean of p(X_t|X_0)

        X_0 * exp{-0.5\int_0^T \beta(t) \varsigma(t)dt}
        """
        mean_coeff = (
            -0.5 * (self._indefinite_int(t) - self._indefinite_int(self.t0))
        ).exp()
        mean = x_t0 * fill_tail_dims(mean_coeff, x_t0)
        return mean

    def analytical_var(self, t, x_t0):
        r"""Compute analytical variance of p(X_t|X_0)

        I - I \exp{-\int_0^T \beta(t) \varsigma(t) dt}
        """
        analytical_var = (
            1 - (-self._indefinite_int(t) + self._indefinite_int(self.t0)).exp()
        )
        return analytical_var

    @torch.no_grad()
    def analytical_sample(self, t, x_t0):
        """Sample Gaussian distribution given

        - mean: computed from `self.analytical_mean`
        - variance: computed from `self.analytical_var`
        """

        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return mean + torch.randn_like(mean) * fill_tail_dims(var.sqrt(), mean)

    @torch.no_grad()
    def analytical_score(self, x_t, t, x_t0):
        r"""As p(X_t|X_0) is a Gaussian distribution $\mathcal{N}(m, v)$

        Its score $\nabla log p(X_t|X_0)$ is

            (x-m) / v
        """
        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return -(x_t - mean) / fill_tail_dims(var, mean).clamp_min(1e-5)

    def f(self, t, y):
        """Modified drift according to white noise

        See Eq. 14 in the paper
        """
        mean_dt, var_sqrt = self.white_noise(t)
        var_sqrt = var_sqrt.clamp(1e-5)

        f = self.original_f(t, y)
        g = self.original_g(t, y)

        ret_f = (
            f * var_sqrt.squeeze().pow(2)
            + g * mean_dt.unsqueeze(-1).unsqueeze(-1) / self.white_noise.dt
        )
        return ret_f

    def g(self, t, y):
        """Modified diffusion according to white noise

        See Eq. 14 in the paper
        """
        _, var_sqrt = self.white_noise(t)
        var_sqrt = var_sqrt.clamp(1e-5)
        g = self.original_g(t, y)
        return g * var_sqrt

    def f_and_g(self, t, y):
        """Modified both drift and diffusion

        See Eq. 14 in the paper
        """
        mean_dt, var_sqrt = self.white_noise(t)
        var_sqrt = var_sqrt.clamp(1e-5)

        f = self.original_f(t, y)
        g = self.original_g(t, y)

        ret_f = f * var_sqrt.pow(2) + g * mean_dt / self.white_noise.dt
        return ret_f, g * var_sqrt

    def sample_t1_marginal(self, batch_size, tau=1.0):
        r"""Sample noise at time T
        P(T) \sim \mathcal{N}(0, I)
        """
        return torch.randn(
            size=(batch_size, *self.input_size), device=self.device
        ) * math.sqrt(tau)

    def lambda_t(self, t):
        """Weight in the loss function"""
        return self.analytical_var(t, None)

    def forward(self, x_t0, partitions=1):
        """Compute the score matching objective.
        Split [t0, t1] into partitions; sample uniformly on each partition to reduce gradient variance.
        """
        u = torch.rand(
            size=(x_t0.shape[0], partitions), dtype=x_t0.dtype, device=x_t0.device
        )
        u.mul_((self.t1 - self.t0) / partitions)
        shifts = torch.arange(0, partitions, device=x_t0.device, dtype=x_t0.dtype)[
            None, :
        ]
        shifts.mul_((self.t1 - self.t0) / partitions).add_(self.t0)
        t = (u + shifts).reshape(-1)
        lambda_t = self.lambda_t(t)

        x_t0 = x_t0.repeat_interleave(partitions, dim=0)

        # resample inducing points and recompute the sample
        self.precompute_white_noise(batch_size=x_t0.size(0))
        x_t = self.analytical_sample(t, x_t0)
        fake_score = self.score(t, x_t)
        true_score = self.analytical_score(x_t, t, x_t0)

        # score matching loss
        loss = lambda_t * ((fake_score - true_score) ** 2).flatten(start_dim=1).sum(
            dim=1
        )
        return loss


class ReverseDiffeqWrapper(Module):
    """
    This class from `torchsde` without any changes
    """

    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, module: ScoreMatchingSDE):
        super(ReverseDiffeqWrapper, self).__init__()
        self.module = module

    # --- odeint ---
    def forward(self, t, y):
        return -(
            self.module.f(-t, y)
            - 0.5 * self.module.g(-t, y) ** 2 * self.module.score(-t, y)
        )

    # --- sdeint ---
    def f(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -(
            self.module.f(-t, y) - self.module.g(-t, y) ** 2 * self.module.score(-t, y)
        )
        return out.flatten(start_dim=1)

    def g(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -self.module.g(-t, y)
        return out.flatten(start_dim=1)

    # --- sample ---
    def sample_t1_marginal(self, batch_size, tau=1.0):
        return self.module.sample_t1_marginal(batch_size, tau)

    @torch.no_grad()
    def ode_sample(self, batch_size=64, tau=1.0, t=None, y=None, dt=1e-2):
        self.module.eval()
        self.module.precompute_white_noise(batch_size=batch_size)
        t = torch.tensor([-self.t1, -self.t0], device=self.device) if t is None else t
        y = self.sample_t1_marginal(batch_size, tau) if y is None else y
        return torchdiffeq.odeint(self, y, t, method="rk4", options={"step_size": dt})

    @torch.no_grad()
    def ode_sample_final(self, batch_size=64, tau=1.0, t=None, y=None, dt=1e-2):
        return self.ode_sample(batch_size, tau, t, y, dt)[-1]

    @torch.no_grad()
    def sde_sample(
        self, batch_size=64, tau=1.0, t=None, y=None, dt=1e-2, tweedie_correction=True
    ):
        self.module.eval()

        t = torch.tensor([-self.t1, -self.t0], device=self.device) if t is None else t
        y = self.sample_t1_marginal(batch_size, tau) if y is None else y
        self.module.precompute_white_noise(batch_size)
        ys = torchsde.sdeint(self, y.flatten(start_dim=1), t, dt=dt)
        ys = ys.view(len(t), *y.size())
        if tweedie_correction:
            ys[-1] = self.tweedie_correction(self.t0, ys[-1], dt)
        return ys

    @torch.no_grad()
    def sde_sample_final(self, batch_size=64, tau=1.0, t=None, y=None, dt=1e-2):
        return self.sde_sample(batch_size, tau, t, y, dt)[-1]

    def tweedie_correction(self, t, y, dt):
        return y + dt**2 * self.module.score(t, y)

    @property
    def t0(self):
        return self.module.t0

    @property
    def t1(self):
        return self.module.t1


def preprocess(x, logit_transform, alpha=0.95):
    if logit_transform:
        x = alpha + (1 - 2 * alpha) * x
        x = (x / (1 - x)).log()
    else:
        x = (x - 0.5) * 2
    return x


def postprocess(x, logit_transform, alpha=0.95, clamp=True):
    if logit_transform:
        x = (x.sigmoid() - alpha) / (1 - 2 * alpha)
    else:
        x = x * 0.5 + 0.5
    return x.clamp(min=0.0, max=1.0) if clamp else x


def make_loader(
    root="./data/mnist",
    train_batch_size=128,
    shuffle=True,
    pin_memory=True,
    num_workers=0,
    drop_last=True,
):
    """Make a simple loader for training images in MNIST."""

    def dequantize(x, nvals=256):
        """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
        return x

    train_transform = tv.transforms.Compose([tv.transforms.ToTensor(), dequantize])
    train_data = tv.datasets.MNIST(
        root, train=True, transform=train_transform, download=True
    )
    train_loader = data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    test_data = tv.datasets.MNIST(
        root, train=False, transform=train_transform, download=False
    )

    test_loader = data.DataLoader(test_data, batch_size=train_batch_size)
    return train_loader, test_loader


# -------------------------------------------------------
#   White noise sampler
# -------------------------------------------------------


class ConstantHurst(SparseGPNoise):
    def __init__(
        self, t0, t1, dt, H=0.3, dim=1, num_steps=100, num_inducings=10
    ) -> None:
        super().__init__(t0, t1, dt, dim, num_steps, num_inducings)
        self.H = H

        # make different inducing points
        del self.Z
        Z_ = torch.linspace(1.1, 2.0, self.num_inducing)
        self.register_buffer("Z", Z_)

    def compute_hurst(self, t):
        t = t.view(-1, 1)
        return torch.ones_like(t) * self.H

    def __repr__(self):
        return f"Fractional Brownian Motion H={self.H}"


class LinearHurstWhiteNoise(SparseGPNoise):
    def __init__(self, t0, t1, dt, ht0=0.3, ht1=0.8, num_steps=100) -> None:
        super().__init__(t0, t1, dt, num_steps=num_steps)
        self.ht0, self.ht1 = ht0, ht1

        # make different inducing points outside of the bound
        del self.Z
        Z_ = torch.linspace(1.1, 2.0, self.num_inducing)
        self.register_buffer("Z", Z_)

    def compute_hurst(self, t):
        ret = self.ht0 + (t - self.t0) * (self.ht1 - self.ht0) / (self.t1 - self.t0)
        ret = ret.view(-1, 1)
        return ret

    def __repr__(self):
        return f"Multifractional Brownian Motion. \
            Linear h(t): ht0={self.ht0}, ht1={self.ht1}, t0,t1={self.t0},{self.t1}"


# -------------------------------------------------------
#   Helper function compute negative log likelihood
# -------------------------------------------------------


def hutch_trace(x_out, x_in, noise=None):
    """Hutchinson's trick computing the trace of a matrix"""
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=False)[0]
    trace = torch.einsum("abcd,abcd->a", jvp, noise)
    return trace


def compute_ll(sde: ScoreMatchingSDE, data, offset, dt=1e-2):
    """Compute log likelihood of a small batch of data"""
    N = np.prod(data.size()[1:])
    data = data.view(-1, N)
    init = torch.cat([torch.zeros(data.size(0), 1).to(data), data], dim=-1)
    t0, t1 = sde.t0, sde.t1
    t = torch.tensor([t0 + 1e-5, t1], device=sde.device)
    noise = torch.randn(size=(data.size(0), *sde.input_size)).to(sde.device)

    def vector_field(t, y):
        with torch.set_grad_enabled(True):
            y_in = y[:, 1:].detach().requires_grad_(True)
            y_in = y_in.view(-1, *sde.input_size)
            sde.precompute_white_noise(batch_size=y_in.size(0))
            y_out = sde.f(t, y_in) - 0.5 * sde.g(t, y_in) * sde.score(t, y_in)
            trace = hutch_trace(y_out, y_in, noise)

        return torch.cat([trace.unsqueeze(-1), y_out.view(-1, N)], dim=-1)

    ys = torchdiffeq.odeint(
        vector_field, init, t, method="euler", options={"step_size": dt}
    )
    ys = ys[-1]  # batch x 785
    delta_lprob = ys[:, 0]
    z = ys[:, 1:]
    lprob = -0.5 * (z**2).sum(-1) - 0.5 * 784 * np.log(2 * np.pi)
    bit_per_dim = -(delta_lprob.squeeze() + lprob.squeeze())
    bit_per_dim = bit_per_dim / z.size(-1) / np.log(2.0)
    bit_per_dim += offset
    # equation 27 in (https://arxiv.org/pdf/1705.07057.pdf)
    last_term = (
        torch.log2(torch.sigmoid(z) + 1e-12)
        + torch.log2(1.0 - torch.sigmoid(z) + 1e-12)
    ).sum(dim=-1) / N
    bit_per_dim += last_term
    return bit_per_dim.sum(dim=0).cpu().data


# -------------------------------------------------------
#   MAIN
# -------------------------------------------------------


def main(
    device="cuda",
    train_dir="./dump/score-based",
    epochs=100,
    lr=1e-4,
    batch_size=128,
    pause_every=1000,
    tau=1.0,
    logit_transform=True,
    alpha=0.05,
    wandb_mode="offline",
):

    os.makedirs(train_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Data.
    train_loader, eval_loader = make_loader(
        root=os.path.join(train_dir, "data"), train_batch_size=batch_size
    )

    # Model + optimizer.
    denoiser = unet.Unet(
        input_size=(1, 28, 28),
        dim_mults=(
            1,
            2,
            4,
        ),
        attention_cls=unet.LinearTimeSelfAttention,
    )

    # TODO: commment and uncomment the following to try out different settings

    # constant Hurst
    # white_noise = ConstantHurst(t0=0.,
    #                             t1=2.1,
    #                             dt=1e-2,
    #                             H=0.3).to(device)

    # linearly increasing Hurst
    # white_noise = LinearHurstWhiteNoise(t0=0.,
    #                                     t1=2.1,
    #                                     dt=1e-2,
    #                                     ht0=0.2,
    #                                     ht1=0.7).to(device)

    # linearly decreasing Hurst
    white_noise = LinearHurstWhiteNoise(t0=0.0, t1=2.1, dt=1e-2, ht0=0.8, ht1=0.1).to(
        device
    )

    forward = ScoreMatchingSDE(denoiser=denoiser, white_noise=white_noise).to(device)
    forward.init()
    reverse = ReverseDiffeqWrapper(forward)
    optimizer = optim.Adam(params=forward.parameters(), lr=lr)

    # equation 27 in (https://arxiv.org/pdf/1705.07057.pdf)
    bpd_offset = -np.log2(1 - 2 * alpha) + 8.0

    x, _ = next(iter(eval_loader))
    x = preprocess(x.to(device), logit_transform=logit_transform, alpha=alpha)
    nll_batch = compute_ll(forward, x, offset=bpd_offset)

    def plot(imgs, path):
        assert not torch.any(torch.isnan(imgs)), "Found nans in images"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imgs = (
            postprocess(imgs, logit_transform=logit_transform, alpha=alpha)
            .detach()
            .cpu()
        )
        tv.utils.save_image(imgs, path)

    wandb.init(
        project="score-based-nll",
        config={"white noise": white_noise.__repr__()},
        mode=wandb_mode,
    )

    global_step = 0
    for epoch in range(epochs):
        for x, _ in tqdm.tqdm(train_loader):
            forward.train()
            forward.zero_grad()
            x = preprocess(x.to(device), logit_transform=logit_transform, alpha=alpha)
            loss = forward(x).mean(dim=0)
            loss.backward()
            optimizer.step()
            global_step += 1
            wandb.log({"loss": loss.detach().cpu().numpy()})

            if global_step % pause_every == 0:
                logging.warning(f"global_step: {global_step:06d}, loss: {loss:.4f}")

                img_path = os.path.join(
                    train_dir, "ode_samples", f"global_step_{global_step:07d}.png"
                )
                ode_samples = reverse.ode_sample_final(tau=tau)
                plot(ode_samples, img_path)
                wandb.log({"ODE-sample": wandb.Image(img_path)})

                img_path = os.path.join(
                    train_dir, "sde_samples", f"global_step_{global_step:07d}.png"
                )
                sde_samples = reverse.sde_sample_final(tau=tau)
                plot(sde_samples, img_path)
                wandb.log({"SDE-sample": wandb.Image(img_path)})

                # get the first batch of eval
                x, _ = next(iter(eval_loader))
                x = preprocess(
                    x.to(device), logit_transform=logit_transform, alpha=alpha
                )
                nll_batch = compute_ll(forward, x, offset=bpd_offset)
                wandb.log({"nll_batch": nll_batch / batch_size})

        if epoch == epochs // 2:
            print("Evaluating negative log likelihood")
            nll = 0.0
            for x, _ in tqdm.tqdm(eval_loader):
                x = preprocess(
                    x.to(device), logit_transform=logit_transform, alpha=alpha
                )
                nll += compute_ll(forward, x, offset=bpd_offset)
            wandb.log({"NLL": nll / 10000.0})

    print("Final evaluating negative log likelihood")
    nll = 0.0
    for x, _ in tqdm.tqdm(eval_loader):
        x = preprocess(x.to(device), logit_transform=logit_transform, alpha=alpha)
        nll += compute_ll(forward, x, offset=bpd_offset)

    wandb.log({"NLL": nll / 10000.0})


if __name__ == "__main__":
    fire.Fire(main)
