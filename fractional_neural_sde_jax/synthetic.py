import math
import os

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from diffrax import (
    ControlTerm,
    diffeqsolve,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    UnsafeBrownianPath,
    VirtualBrownianTree,
    WeaklyDiagonalControlTerm,
)
from fractional_neural_sde_jax.fractional_sde import convert_sde
from fractional_neural_sde_jax.sde_kl_divergence import sde_kl_divergence
from fractional_neural_sde_jax.sparse_gp import FractionalSparseGP
from tqdm import tqdm


class DriftPosterior(eqx.Module):

    net: eqx.nn.MLP

    def __init__(self, *, key) -> None:

        self.net = eqx.nn.MLP(
            in_size=3,
            width_size=100,
            out_size=1,
            depth=3,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        input = jnp.concatenate([jnp.sin(t[None]), jnp.cos(t[None]), y])
        return self.net(input)


class DriftPrior(eqx.Module):

    theta: float
    mu: float

    def __call__(self, t, y, args):
        return self.theta * (self.mu - y)


class Diffusion(eqx.Module):

    sigma: float

    def __call__(self, t, y, args):
        return jnp.ones_like(y) * self.sigma


class Hurst(eqx.Module):

    mlp: eqx.nn.MLP
    positional_encoding: bool

    def __init__(
        self,
        dim=1,
        positional_encoding: bool = True,
        *,
        key: "jax.random.PRNGKey",  # noqa: F821
    ) -> None:
        self.positional_encoding = positional_encoding
        if positional_encoding:
            in_size = 3
        else:
            in_size = 1
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=dim,
            width_size=10,
            depth=1,
            activation=jax.nn.tanh,
            final_activation=jax.nn.sigmoid,
            key=key,
        )

    def __call__(self, t):
        if t.ndim == 0:
            t = t[None]
        if self.positional_encoding:
            t = jnp.concatenate([jnp.sin(t), jnp.cos(t), t])
        return self.mlp(t)


def laplace_logprob(y, loc, scale):
    return -jnp.log(2 * scale) - jnp.abs(y - loc) / scale


def normal_logprob(y, loc, scale):
    return -0.5 * ((y - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)


def normal_kl_divergence(loc1, scale1, loc2, scale2):
    return 0.5 * (
        (scale1 / scale2) ** 2
        + (loc2 - loc1) ** 2 / scale2**2
        - 1
        + jnp.log(scale2 / scale1)
    )


class LatentSDE(eqx.Module):

    drift_posterior: eqx.Module
    drift_prior: eqx.Module
    diffusion: eqx.Module
    hurst_fn: eqx.Module
    sparse_gp: FractionalSparseGP
    qy0_mean: jnp.ndarray
    qy0_logvar: jnp.ndarray
    py0_mean: float
    py0_logvar: float
    t0: float
    t1: float

    def __init__(
        self, hurst_fn, drift_posterior, t0, t1, dt, theta=1.0, mu=1.0, sigma=0.5
    ) -> None:
        super().__init__()

        self.hurst_fn = hurst_fn
        self.sparse_gp = FractionalSparseGP(hurst_fn, t0=t0, t1=t1, dt=dt)

        # specify drift and diffusion
        self.drift_posterior = drift_posterior
        # self.drift_posterior = drift_posterior
        self.drift_prior = DriftPrior(theta, mu)
        self.diffusion = Diffusion(sigma)

        # define distribution at the initial point
        self.qy0_mean = jnp.array([mu])
        self.qy0_logvar = jnp.log(jnp.array([sigma]) ** 2 / (2.0 * theta))
        self.py0_mean = mu
        self.py0_logvar = math.log(sigma**2 / (2 * theta))

        self.t0, self.t1 = t0, t1

    @property
    def py0_std(self):
        return math.exp(0.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return jnp.exp(0.5 * self.qy0_logvar)

    def solve(self, y0, solver, dt=1e-2, saveat=None, *, key):

        bm_key, inducing_key = jrandom.split(key)
        bm = VirtualBrownianTree(
            t0=self.t0, t1=self.t1, shape=(1,), tol=1e-3, key=bm_key
        )
        new_drift, new_diffusion, init_fn = convert_sde(
            drift=self.drift_posterior,
            diffusion=self.diffusion,
            sparse_gp=self.sparse_gp,
        )
        control_term = WeaklyDiagonalControlTerm(new_diffusion, bm)
        posterior_drift = ODETerm(new_drift)
        prior_drift = ODETerm(self.drift_prior)
        aug_sde, aug_y0 = sde_kl_divergence(
            drift1=posterior_drift, drift2=prior_drift, diffusion=control_term, y0=y0
        )

        inducing_info = init_fn(1, inducing_key)
        sol = diffeqsolve(
            aug_sde,
            solver,
            t0=self.t0,
            t1=self.t1,
            dt0=dt,
            y0=aug_y0,
            args=(None,)
            + inducing_info,  # first args is for context, there is no context here
            saveat=saveat,
        )
        return sol.ys

    def __call__(self, ts, batch_size, *, key):

        eps_key, bm_key = jrandom.split(key)
        solver = Euler()
        saveat = SaveAt(ts=ts, dense=True)

        def solve(y0, key):
            ys, logpq = self.solve(y0, solver=solver, saveat=saveat, key=key)
            return ys, logpq

        eps = jrandom.normal(
            key=eps_key,
            shape=(
                batch_size,
                1,
            ),
        )
        y0 = self.qy0_mean + eps * self.qy0_std
        logpq0 = normal_kl_divergence(
            loc1=self.qy0_mean,
            scale1=self.qy0_std,
            loc2=self.py0_mean,
            scale2=self.py0_std,
        )

        batch_solve = jax.vmap(solve)
        bm_key = jrandom.split(bm_key, batch_size)
        ys, logqp_path = batch_solve(y0, bm_key)
        logpq = jnp.mean(logpq0 + logqp_path[-1])

        return ys, logpq

    def sample(self, drift, mean0, std0, batch_size=1024, dt=1e-2, *, key):

        bm_key, eps_key = jrandom.split(key)
        ts_vis = jnp.linspace(self.t0, self.t1, 100)
        solver = Euler()
        saveat = SaveAt(ts=ts_vis, dense=True)

        def solve(y0, key):
            bm = VirtualBrownianTree(
                t0=self.t0, t1=self.t1, tol=1e-3, shape=(1,), key=key
            )
            sde = MultiTerm(ODETerm(drift), ControlTerm(self.diffusion, bm))
            sol = diffeqsolve(
                sde, solver=solver, t0=self.t0, t1=self.t1, dt0=dt, y0=y0, saveat=saveat
            )
            return sol.ys

        eps = jrandom.normal(key=eps_key, shape=(batch_size, 1))
        y0 = mean0 + std0 * eps
        ys_vis = jax.vmap(solve)(y0, key=jrandom.split(bm_key, batch_size))

        return np.squeeze(np.array(ts_vis)), np.squeeze(np.array(ys_vis))


def create_data(t0=0, t1=2, dt=1e-2, alpha=0.5, beta=0.5, x0=1.0, *, key):
    inducing_key, bm_key = jrandom.split(key)

    def true_hurst_fn(t):
        return jax.nn.sigmoid((1 - t) * 7.0) * 0.5 + 0.3

    sparse_gp = FractionalSparseGP(
        hurst_fn=true_hurst_fn,
        t0=t0,
        t1=t1,
        dt=dt,
        num_steps=500,
        num_inducings=70,
    )

    args = sparse_gp.sample_inducing(1, inducing_key)
    bm = UnsafeBrownianPath(key=bm_key, shape=(1,))

    n_samples = int((t1 - t0) / dt)
    ts = jnp.linspace(t0, t1, n_samples)

    delta_bs = []
    for ta, tb in zip(ts[:-1], ts[1:]):
        mean, std = sparse_gp(ta, *args)
        delta_b = mean * dt + std * bm.evaluate(ta, tb)
        delta_bs += [delta_b]

    delta_bs = jnp.concatenate(delta_bs)
    delta_bs = jnp.append(0.0, delta_bs)
    path = jnp.cumsum(delta_bs)

    ht = jax.vmap(true_hurst_fn)(ts)
    xt = x0 * jnp.exp(beta * path + alpha * ts - 0.5 * beta**2 * ts ** (2 * ht))
    return ts, xt, ht


def plot_posterior(model: LatentSDE, ts, ys, file_name, *, key):

    # set up plotting variables
    palette = sns.color_palette("Blues_r")
    plt.figure(figsize=(6.5, 4))
    num_samples = 3
    sample_colors = [palette[i + 1] for i in range(num_samples)]
    fill_color = palette[2]
    mean_color = palette[0]
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # get samples
    ts_vis, ys_vis = model.sample(
        drift=model.drift_posterior, mean0=model.qy0_mean, std0=model.qy0_std, key=key
    )

    # plot confidence intervals
    ys_vis_ = np.sort(ys_vis, axis=0)
    for alpha, percentile in zip(alphas, percentiles):
        idx = int((1 - percentile) / 2.0 * ys_vis.shape[0])
        ys_bot, ys_top = ys_vis_[idx], ys_vis_[-idx]
        plt.fill_between(ts_vis, ys_bot, ys_top, alpha=alpha, color=fill_color)

    # plot mean
    plt.plot(
        ts_vis, ys_vis.mean(axis=0), color=mean_color, linestyle="--", linewidth=2.5
    )

    # plot samples
    for j in range(num_samples):
        plt.plot(ts_vis, ys_vis[j], color=sample_colors[j], linewidth=1.5)

    # plot arrows
    num, dt = 12, 0.12
    t, y = jnp.meshgrid(jnp.linspace(0.2, 1.8, num), jnp.linspace(-1.5, 1.5, num))
    t, y = t.reshape((-1,)), y.reshape((-1, 1))
    vectors = jax.vmap(model.drift_posterior)(t, y, None)
    dt = np.ones((num, num)) * dt
    dy = vectors.reshape((num, num)) * dt
    plt.quiver(t, y, dt, dy, alpha=0.3, edgecolor="k", width=0.0035, scale=50)

    # plot data
    plt.scatter(ts, ys, marker="x", zorder=3, color="k", s=50)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$X_t$")
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_h(hurst_fn, ts, true_ht, shift, file_name):
    """Plot Hurst function"""
    plt.figure(figsize=(5, 4))
    palette = sns.color_palette()

    ht = jax.vmap(hurst_fn)(ts)

    plt.plot(ts, ht, label="ours", color=palette[0], alpha=0.7)
    plt.plot(ts, true_ht, label="true", color=palette[2], alpha=0.7)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$h(t)$")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True)
    plt.ylim([0.2, 1])
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    train_dir="./dump/",
    t0=0.0,
    t1=2.0,
    dt=5 * 1e-3,
    lr=1e-2,
    train_iters=5000,
    kl_anneal_iters=100,
    batch_size=1024,
    logprob_fn=laplace_logprob,
    scale=0.025,
    print_freq=100,
    seed=0,
):
    train_dir = os.path.join(train_dir, "synthetic_jax")
    os.makedirs(train_dir, exist_ok=True)

    key = jrandom.PRNGKey(seed)
    data_key, posterior_key, hurst_key, sde_key, vis_key = jrandom.split(key, 5)

    ts, ys, ht = create_data(
        t0=t0,
        t1=t1,
        key=data_key,
    )

    shift = 0.1
    ts = ts + shift

    drift_posterior = DriftPosterior(key=posterior_key)
    hurst_fn = Hurst(dim=1, key=hurst_key)
    latent_sde = LatentSDE(
        hurst_fn=hurst_fn, drift_posterior=drift_posterior, t0=t0, t1=t1 + shift, dt=dt
    )

    schedule = optax.exponential_decay(
        init_value=lr,
        decay_rate=0.999,
        transition_steps=1,
    )
    optim = optax.adam(learning_rate=schedule)
    opt_state = optim.init(eqx.filter(latent_sde, eqx.is_array))

    iter = 0

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def make_step(model):
        ys_pred, kl = model(ts, batch_size=batch_size, key=key)
        log_likelihood = logprob_fn(
            y=jnp.squeeze(ys_pred), loc=jnp.squeeze(ys), scale=scale
        )
        log_likelihood = jnp.mean(jnp.sum(log_likelihood, axis=-1), axis=-1)
        return -log_likelihood + kl * min(1.0, (iter + 1) / kl_anneal_iters)

    pbar = tqdm(total=train_iters)
    while iter < train_iters:
        # optimizing
        loss, grads = make_step(latent_sde)
        sde_key = jrandom.split(sde_key, 1)[0]
        loss = loss.item()
        updates, opt_state = optim.update(grads, opt_state)
        latent_sde = eqx.apply_updates(latent_sde, updates)
        if iter % print_freq == 0:
            print(f"Iteration {iter} \t Loss: {loss:.3f}")
            posterior_file = os.path.join(train_dir, f"step_{iter}.png")
            plot_posterior(latent_sde, ts, ys, file_name=posterior_file, key=vis_key)
            h_file = os.path.join(train_dir, f"h_{iter}.png")
            plot_h(hurst_fn, ts, ht, shift=shift, file_name=h_file)
            plt.show()
        iter += 1
        pbar.update(1)


if __name__ == "__main__":
    fire.Fire(main)
