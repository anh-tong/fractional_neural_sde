from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy
from diffrax.custom_types import Array, Scalar


_matmul1 = jax.vmap(jnp.matmul)
_matmul2 = jax.vmap(_matmul1)
_cholesky = jax.vmap(jnp.linalg.cholesky)


def _cholesky_solve(A, b):
    def solve(_A, _b):
        return jax.scipy.linalg.cho_solve((_A, True), _b)

    return jax.vmap(solve)(A, b)


def compute_kernel(phi_1, phi_2=None):
    """

    Args:
        phi_1: size (num_steps, dim, size_1)
        phi_2: size (num_steps, dim, size_2). Default phi_2 = phi_1

    Returns:
        kernel function with size (dim, size_1, size_2)
    """
    if phi_2 is None:
        phi_2 = phi_1

    phi_1 = phi_1[..., None]
    phi_2 = jnp.swapaxes(phi_2[..., None], -1, -2)
    kernel = _matmul2(phi_1, phi_2).sum(axis=0)
    return kernel


class FractionalSparseGP:
    def __init__(
        self,
        hurst_fn: Callable,
        t0: Scalar,
        t1: Scalar,
        dt: Scalar,
        num_steps: int = 100,
        num_inducings: int = 70,
        inducing_loc: Array = None,
    ) -> None:
        """
        Sparse GP for fractional Brownian Motions

        Args:
            hurst_fn: a function return a value of Hurst exponent h(t)
            t0: the starting time
            t1: the terminal time
            dt: fixed step size of SDE solver
            num_inducings: number of inducing points
            inducing_loc: location of inducing points
        """

        self.hurst_fn = hurst_fn

        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.steps = jnp.linspace(t0, t1, num_steps + 1)
        self.ds = (t1 - t0) / num_steps

        # initialize inducing location
        if inducing_loc is None:
            self.num_inducings = num_inducings
            self.inducing_loc = jnp.linspace(
                t0 + dt * 5.5,
                t1 - dt * 5.5,
                self.num_inducings,
            )
        else:
            self.inducing_loc = inducing_loc
            self.num_inducings = inducing_loc.shape[0]

    def sample_inducing(
        self,
        batch_size,
        key: "jax.random.PRNGKey",  # noqa: F821
    ) -> Tuple[Array]:

        phi_inducing = jax.vmap(self.compute_phi)(self.inducing_loc)
        phi_inducing = jnp.transpose(phi_inducing, (1, 2, 0))
        kernel_inducing = compute_kernel(phi_inducing) + 1e-6 * jnp.eye(
            phi_inducing.shape[-1]
        )
        chol_inducing = _cholesky(kernel_inducing)
        eps = jrandom.normal(
            key=key, shape=(chol_inducing.shape[0], chol_inducing.shape[1], batch_size)
        )
        inducing_value = jax.vmap(jnp.matmul)(chol_inducing, eps)

        return inducing_value, phi_inducing, chol_inducing

    def integral_weight(self, t) -> Array:
        """
        Compute integral weight

        See g_i(t) in the paper.

        Args:
            t: input time

        Returns:
            an array with size (`self.num_steps`, output dim of `hurst_fn`)
        """
        if t.ndim == 0:
            t = t[None]
        h = self.hurst_fn(t)[None, ...]
        alpha = 2 * h

        t_t0 = jax.nn.relu(t - self.steps[:-1])[..., None]
        t_t1 = jax.nn.relu(t - self.steps[1:])[..., None]

        weight = jnp.sqrt(t_t0**alpha - t_t1**alpha) / jnp.sqrt(alpha * self.ds)
        weight = weight / jnp.exp(jax.lax.lgamma(h + 0.5))

        return weight

    def compute_phi(self, t) -> Array:
        """Compute the feature w.r.t. kernel

        Args:
            t: input time

        Returns:
            an array with size = `self.num_steps`
        """

        ta, tb = t, t + self.dt
        wa, wb = self.integral_weight(ta), self.integral_weight(tb)
        phi = (wb - wa) * jnp.sqrt(self.ds) / self.dt

        return phi

    def compute_mean(
        self, t, inducing_value: Array, phi_inducing: Array, chol_inducing: Array
    ) -> Array:
        r"""
        Compute mean of sparse GP at time t

        Args:
            t : input time
            inducing_value (Array): inducing point values. This is equivalent to \Delta B_Z
            phi_inducing (Array): feature w.r.t evaluated at inducing point
            chol_inducing (Array): Cholesky matrix w.r.t to the kernel matrix evaluated at inducing points

        Returns:
            A tuple contains
                - mean: size (batch_size, dim)
        """
        phi_t = self.compute_phi(t)[..., None]
        kernel_tz = compute_kernel(phi_inducing, phi_t)
        alpha = _cholesky_solve(chol_inducing, kernel_tz)

        mean = _matmul1(jnp.swapaxes(alpha, -2, -1), inducing_value)
        mean = jnp.reshape(mean, (inducing_value.shape[-1], inducing_value.shape[0]))

        return mean

    def compute_std(self, t, phi_inducing: Array, chol_inducing: Array) -> Array:
        r"""
        Compute standard dev. and multiplied with squared root of `dt`

        Args:
            t : input time
            phi_inducing (Array): feature w.r.t evaluated at inducing point
            chol_inducing (Array): Cholesky matrix w.r.t to the kernel matrix evaluated at inducing points

        Returns:
            A tuple contains
                - standard dev. multiplied with squared root of `dt` (SDE solver's step size): size (1, dim)
        """
        phi_t = self.compute_phi(t)[..., None]
        kernel_tz = compute_kernel(phi_inducing, phi_t)
        kernel_tt = compute_kernel(phi_t)
        alpha = _cholesky_solve(chol_inducing, kernel_tz)

        var = kernel_tt - _matmul1(jnp.swapaxes(alpha, -2, -1), alpha)
        var = jnp.reshape(var, (1, var.shape[0]))

        return jnp.sqrt(jnp.clip(var, a_min=0) * self.dt)

    def __call__(
        self, t, inducing_value: Array, phi_inducing: Array, chol_inducing: Array
    ) -> Tuple[Array]:
        r"""
        Compute mean and variance of sparse GP at time t

        Args:
            t : input time
            inducing_value (Array): inducing point values. This is equivalent to \Delta B_Z
            phi_inducing (Array): feature w.r.t evaluated at inducing point
            chol_inducing (Array): Cholesky matrix w.r.t to the kernel matrix evaluated at inducing points

        Returns:
            A tuple contains
                - mean: size (batch_size, dim)
                - standard dev. multiplied with squared root of `dt` (SDE solver's step size): size (1, dim)
        """

        phi_t = self.compute_phi(t)[..., None]
        kernel_tz = compute_kernel(phi_inducing, phi_t)
        kernel_tt = compute_kernel(phi_t)
        alpha = _cholesky_solve(chol_inducing, kernel_tz)

        mean = _matmul1(jnp.swapaxes(alpha, -2, -1), inducing_value)
        var = kernel_tt - _matmul1(jnp.swapaxes(alpha, -2, -1), alpha)

        mean = jnp.reshape(mean, (inducing_value.shape[-1], inducing_value.shape[0]))
        var = jnp.reshape(var, (1, var.shape[0]))

        return (mean, jnp.sqrt(jnp.clip(var, a_min=0) * self.dt))
