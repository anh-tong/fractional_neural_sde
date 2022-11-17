import math
import os
from functools import partial

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax
import torchvision as tv
from diffrax import (
    Bosh3,
    ControlTerm,
    diffeqsolve,
    Euler,
    MultiTerm,
    NoAdjoint,
    ODETerm,
    UnsafeBrownianPath,
)
from fractional_neural_sde_jax.sparse_gp import FractionalSparseGP
from fractional_neural_sde_jax.unet import UNet2DModel
from torch.utils import data
from tqdm.auto import tqdm


class Process:
    def mean(self, *args):
        pass

    def variance(self, *args):
        pass

    def sample(self, *args):
        pass


class Forward(Process):
    def mean(self, x0, t):
        """Compute the MEAN the forward process
        q(X_t| X_0)
        """
        raise NotImplementedError

    def variance(self, t):
        """Compute the VARIANCE of the forward process
            q(X_t| X_0)
        This function does not depend on X_0
        """
        raise NotImplementedError

    def sample(self, x0, t, *, key):
        mean = self.mean(x0, t)
        var = self.variance(t)
        std = jnp.sqrt(var.clip(min=0))
        noise = jrandom.normal(key=key, shape=x0.shape)
        return mean + std * noise


class Backward(Process):
    def __init__(self, forward: Forward):

        self.forward = forward

    def mean(self, x, t, score_fn, *args):
        raise NotImplementedError

    def variance(self, t):
        raise NotImplementedError

    def sample(self, x, t, score_fn, *args):
        raise NotImplementedError


class ContinuousForward(Forward):

    r"""Continuous version of forward process
    Time t \in [0, 1]
    SDE:
        dX_t = -0.5 beta_t X_t dt + sqrt(beta_t) dW_t
    """

    def __init__(
        self,
        beta_min,
        beta_max,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def beta_int(self, t):
        return self.beta_min * t + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def drift(self, x, t, *args):
        return -0.5 * self.beta(t, *args) * x

    def diffusion(self, t, *args):
        return jnp.sqrt(self.beta(t, *args))

    def mean(self, x0, t, *args):
        return jnp.exp(-0.5 * self.beta_int(t, *args)) * x0

    def variance(self, t, *args):
        return 1.0 - jnp.exp(-self.beta_int(t, *args))

    def sample(
        self,
        x0,
        t,
        *args,
        key,
    ):
        mean_t = self.mean(x0, t, *args)
        var_t = self.variance(t, *args)
        std_t = jnp.sqrt(var_t.clip(min=0.0))
        noise = jrandom.normal(key=key, shape=x0.shape)

        return mean_t + std_t * noise

    def compute_loss(self, score_fn, x0, n_steps=1000, *, key):
        """
        Compute score matching loss
        Args:
            score_fn: callable function
            x0: original input (image)
            n_steps: number of discretized steps
            key: jax random generator key
        """
        t_key, noise_key = jrandom.split(key)
        t = jrandom.randint(key=t_key, shape=(1,), minval=1, maxval=n_steps) / (
            n_steps - 1
        )

        noise = jrandom.normal(key=noise_key, shape=x0.shape)

        # sample X_t based on forward process
        mean_t = self.mean(x0, t)
        var_t = jnp.maximum(self.variance(t), 1e-5)
        std_t = jnp.sqrt(var_t)
        x_t = mean_t + std_t * noise

        fake_score = score_fn(x_t, t)
        true_score = -noise / std_t
        weight = var_t
        loss = weight * (fake_score - true_score) ** 2
        return jnp.mean(loss)


class ContinuousBackward(Backward):
    r"""
    Backward (reverse) SDE:
        dX_t = -0.5 beta_t * X_t - beta_t \nabla_x log p(X,t) + \sqrt(beta_t)dW_t
    """

    def __init__(
        self,
        forward: ContinuousForward,
    ):
        super().__init__(forward)

    def mean(
        self,
    ):
        pass

    def drift(self, x, t, score_fn, *args):
        return self.forward.drift(x, t, *args) - self.forward.diffusion(
            t, *args
        ) ** 2 * score_fn(x, t)

    def diffusion(self, t, *args):
        return self.forward.diffusion(t, *args)

    def sample_step(self, x, t, dt, score_fn, *, key):
        drift = self.drift(x, t, score_fn)
        diffusion = self.diffusion(t)
        noise = jrandom.normal(key=key, shape=x.shape)
        return drift * dt + diffusion * jnp.sqrt(dt) * noise

    def sample(self, score_fn, shape, n_steps=1000, ode=True, *, key):
        """Generate image from noise"""
        if ode:

            def drift(t, y, args):
                return self.forward.drift(y, t) - 0.5 * self.forward.diffusion(
                    t
                ) ** 2 * score_fn(y, t)

            term = ODETerm(drift)
            noise_key = key
        else:

            def drift(t, y, args):
                r"""
                f(X_t,t) - g(t)**2 * \nabla_x \log p(X_t, t)
                """
                return self.drift(y, t, score_fn)

            def diffusion(t, y, args):
                return self.diffusion(t) * jnp.ones_like(y)

            noise_key, path_key = jrandom.split(key)
            drift_term = ODETerm(drift)
            diffusion_term = ControlTerm(
                diffusion,
                UnsafeBrownianPath(key=path_key, shape=shape),
            )
            term = MultiTerm(drift_term, diffusion_term)

        solver = Euler()
        y1 = jrandom.normal(key=noise_key, shape=shape)

        sol = diffeqsolve(
            term,
            solver=solver,
            t0=1.0,
            t1=0.0,
            dt0=-1.0 / (n_steps - 1),
            y0=y1,
            adjoint=NoAdjoint(),
        )

        return sol.ys[0]


class FractionalForward(ContinuousForward):
    def __init__(
        self, sparse_gp: FractionalSparseGP, beta_min, beta_max, t0=0.0, t1=1.0
    ) -> None:
        super().__init__(beta_min, beta_max)
        self.sparse_gp = sparse_gp
        self.t0, self.t1 = t0, t1

        # discretization mesh
        self.steps = jnp.linspace(self.t0, self.t1, 100)
        self.ds = self.steps[1] - self.steps[0]
        self.evaluated_beta = jnp.squeeze(self.beta(self.steps))

    def initialize(self, key):
        inducing_info = self.precompute_white_noise(key=key)
        std = jax.vmap(lambda t: self.sparse_gp.compute_std(t, *inducing_info[1:]))(
            self.steps
        )
        self.var = jnp.squeeze(std) ** 2

    def precompute_white_noise(self, batch_size=1, *, key):
        inducing_info = self.sparse_gp.sample_inducing(batch_size=batch_size, key=key)
        return inducing_info

    def beta_int(self, t, *args):
        """Note that that this function no longer serve as interation of beta"""
        diff = t - self.steps
        indicator = jnp.greater(diff, 0).astype(jnp.float32)
        indicator = jnp.squeeze(indicator)
        ret = indicator * self.evaluated_beta * self.var
        ret = jnp.sum(ret) * self.ds

        return ret

    def drift(self, x, t, *args):
        inducing_info = args
        mean, std = self.sparse_gp(t, *inducing_info)
        original_drift = super().drift(x, t)
        original_diffusion = super().diffusion(t)
        # beta_t = self.beta(t)
        # ret = -0.5 * beta_t * x * (std ** 2) + jnp.sqrt(beta_t) * std * mean
        ret = original_drift * std**2 + original_diffusion * mean
        return ret

    def diffusion(self, t, *args):
        inducing_info = args
        std = self.sparse_gp.compute_std(t, *inducing_info[1:])
        original_diffusion = super().diffusion(t)
        return original_diffusion * std

    def compute_loss(self, score_fn, x0, n_steps=1000, *args, key):

        t_key, noise_key = jrandom.split(key, 2)
        t = jrandom.randint(key=t_key, shape=(1,), minval=1, maxval=n_steps) / (
            n_steps - 1
        )

        noise = jrandom.normal(key=noise_key, shape=x0.shape)

        # sample X_t based on forward process
        mean_t = self.mean(x0, t, *args)
        var_t = jnp.maximum(self.variance(t, *args), 1e-5)
        std_t = jnp.sqrt(var_t)
        x_t = mean_t + std_t * noise

        fake_score = score_fn(x_t, t)
        true_score = -noise / std_t
        weight = var_t
        loss = weight * (fake_score - true_score) ** 2
        return jnp.sum(loss)


class FractionalBackward(ContinuousBackward):
    def __init__(self, forward: FractionalForward):
        super().__init__(forward)

    def sample(self, score_fn, shape, dt, ode=True, tweedie_correction=True, *, key):
        """Generate image from noise"""
        if ode:

            def drift(t, y, args):
                return self.forward.drift(y, t, *args) - 0.5 * self.forward.diffusion(
                    t, *args
                ) ** 2 * score_fn(y, t)

            term = ODETerm(drift)
            noise_key = key
        else:

            def drift(t, y, args):
                return self.drift(y, t, score_fn, *args)

            def diffusion(t, y, args):
                return self.diffusion(t, *args) * jnp.ones_like(y)

            noise_key, path_key = jrandom.split(key)
            drift_term = ODETerm(drift)
            diffusion_term = ControlTerm(
                diffusion,
                UnsafeBrownianPath(key=path_key, shape=shape),
            )
            term = MultiTerm(drift_term, diffusion_term)

        solver = Bosh3() if ode else Euler()
        noise_key, inducing_key = jrandom.split(noise_key)
        inducing_info = self.forward.precompute_white_noise(key=inducing_key)
        y1 = jrandom.normal(key=noise_key, shape=shape)
        sol = diffeqsolve(
            term,
            solver=solver,
            t0=1.0,
            t1=0.0,
            dt0=-dt,
            y0=y1,
            args=inducing_info,
            adjoint=NoAdjoint(),
        )

        ret = sol.ys[0]
        if tweedie_correction:
            ret = ret + dt * score_fn(ret, jnp.array(0.0))
        return ret


def numpy_collate(batch):
    """This is taken from JAX doc"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    """This is taken from JAX doc"""

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def make_loader(
    root="./data/mnist",
    train_batch_size=128,
    shuffle=True,
    pin_memory=True,
    num_workers=0,
    drop_last=True,
):
    def cast(pic):
        return np.array(pic, dtype=jnp.float32)

    def dequantize(x, nvals=256):
        x = x * (nvals - 1)
        x = x / nvals
        return x

    train_transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            dequantize,
            cast,
        ]
    )

    train_data = tv.datasets.MNIST(
        root,
        train=True,
        transform=train_transform,
        download=True,
    )

    train_data_loader = NumpyLoader(
        train_data,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return train_data_loader


def preprocess(x, logit_transform=True, alpha=0.05):
    if logit_transform:
        x = alpha + (1 - 2 * alpha) * x
        x = jnp.log(x / (1 - x))
    else:
        x = (x - 0.5) * 2
    return x


def postprocess(x, logit_transform=True, alpha=0.05, clip=True):
    if logit_transform:
        x = (jax.nn.sigmoid(x) - alpha) / (1 - 2 * alpha)
    else:
        x = x * 0.5 + 0.5
    return jnp.clip(x, 0.0, 1.0) if clip else x


def plot_forward(forward_sample, sample, file_dir, ts=[0.3, 0.6, 1.0], *, key):
    sample_size2 = sample.shape[0]
    sample_size = math.floor(math.sqrt(sample_size2))
    sample = sample[: sample_size**2]
    sample = preprocess(sample)
    for t in ts:
        keys = jrandom.split(key, sample_size**2)
        sample_fn = lambda x, k: forward_sample(x, t, key=k)
        generated = jax.vmap(sample_fn)(sample, keys)
        generated = einops.rearrange(
            generated,
            "(n1 n2) 1 h w -> (n1 h) (n2 w)",
            n1=sample_size,
            n2=sample_size,
        )

        plt.figure()
        plt.imshow(generated, cmap="Greys")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, f"forward_{t}.png"))
        plt.close()


def plot_backward(generate_fn, file_name, sample_size=10, *, key):
    sample_key = jrandom.split(key, sample_size**2)
    generated = generate_fn(sample_key)
    generate_fn = postprocess(generated)
    sample = einops.rearrange(
        generated,
        "(n1 n2) 1 h w -> (n1 h) (n2 w)",
        n1=sample_size,
        n2=sample_size,
    )

    plt.imshow(sample, cmap="Greys")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def main(
    train_dir="./dump/score-based-jax",
    use_fractional=True,
    epochs=300,
    lr=3 * 1e-4,
    batch_size=256,
    pause_every=2,
    logit_transform=True,
    dt=1e-2,
    alpha=0.05,
    seed=123,
):
    os.makedirs(train_dir, exist_ok=True)

    key = jrandom.PRNGKey(seed)
    model_key, train_key, sample_key = jrandom.split(key, 3)
    train_loader = make_loader(
        root=os.path.join(train_dir, "data"),
        train_batch_size=batch_size,
    )

    model = UNet2DModel(
        in_channels=1,
        out_channels=1,
        block_out_channels=(64, 128, 256),
        layers_per_block=2,
        attn_head_dim=32,
        key=model_key,
    )

    if use_fractional:
        # hurst will be linear interpolation between `a` and `b`
        # here, we extend the time range of sparse GP (double that of SDE)
        # inducing locations should be chosen after the terminal time of SDE.
        t0, t1 = 0.0, 2.1
        a, b = 0.8, 0.2
        hurst_fn = lambda t: a + (t - t0) * (b - a) / (t1 - t0)
        inducing_loc = jnp.linspace(1.1, 2.0, 10)
        sparse_gp = FractionalSparseGP(
            hurst_fn=hurst_fn,
            t0=t0,
            t1=t1,
            dt=dt,
            inducing_loc=inducing_loc,
        )
        forward = FractionalForward(
            sparse_gp=sparse_gp,
            beta_min=0.1,
            beta_max=50.0,
            t0=0.0,
            t1=1.0,
        )
        model_key, init_key = jrandom.split(model_key)
        forward.initialize(key=init_key)
        backward = FractionalBackward(forward)
    else:
        forward = ContinuousForward(beta_min=0.1, beta_max=10.0)
        backward = ContinuousBackward(forward)

    # plot samples of forward SDE
    images, _ = next(iter(train_loader))
    plot_forward(
        forward.sample, sample=images[:100], file_dir=train_dir, key=sample_key
    )

    optim = optax.adam(learning_rate=lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    def _single_loss_fn(score_fn, x0, key):
        x0 = preprocess(x0, logit_transform, alpha)
        return forward.compute_loss(score_fn, x0, 100, key=key)

    @eqx.filter_value_and_grad
    def batch_loss_fn(score_fn, data, key):
        single_loss_fn = partial(_single_loss_fn, score_fn)
        batch_loss = jax.vmap(single_loss_fn)
        loss = batch_loss(data, jrandom.split(key, data.shape[0]))
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(score_fn, data, key, opt_state, opt_update):

        loss, grads = batch_loss_fn(score_fn, data, key)
        updates, opt_state = opt_update(grads, opt_state)
        score_fn = eqx.apply_updates(score_fn, updates)
        key = jrandom.split(key, 1)[0]
        return loss, score_fn, key, opt_state

    epoch_pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in epoch_pbar:
        pbar = tqdm(train_loader, position=0, leave=True)
        loss = 0.0
        total_loss = 0.0
        for x, _ in pbar:
            loss, model, train_key, opt_state = make_step(
                score_fn=model,
                data=x,
                key=train_key,
                opt_state=opt_state,
                opt_update=optim.update,
            )
            loss = loss.item()
            total_loss += loss
            pbar.set_postfix({"Loss": loss, "Total loss": total_loss})

        epoch_pbar.set_postfix({"Total loss": total_loss})

        if epoch % pause_every == 0:
            img_path = os.path.join(train_dir, f"epoch_{epoch}.png")
            generate_fn = jax.jit(
                jax.vmap(
                    lambda k: backward.sample(
                        model, shape=x.shape[1:], dt=dt, ode=True, key=k
                    )
                )
            )
            plot_backward(generate_fn, img_path, sample_size=8, key=sample_key)


if __name__ == "__main__":
    main()
