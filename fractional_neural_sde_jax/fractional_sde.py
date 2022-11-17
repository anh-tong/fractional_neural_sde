import jax.numpy as jnp
from fractional_neural_sde_jax.sparse_gp import FractionalSparseGP


def convert_sde(
    drift,
    diffusion,
    sparse_gp: FractionalSparseGP,
):
    """Convert Fractional SDE to SDE with Brownian motions

    See Eq. 12 in the paper
    """
    init_fn = sparse_gp.sample_inducing

    def new_drift(t, y, args):

        # passing inducing point info via `args`
        inducing_value, phi_inducing, chol_inducing = args[:3]
        mean = sparse_gp.compute_mean(t, inducing_value, phi_inducing, chol_inducing)

        return drift(t, y, args=None) + jnp.squeeze(mean) * diffusion(t, y, args=None)

    def new_diffusion(t, y, args):

        # passing inducing point info via `args`
        _, phi_inducing, chol_inducing = args[:3]
        std_dt = sparse_gp.compute_std(t, phi_inducing, chol_inducing)

        return diffusion(t, y, args=None) * jnp.squeeze(std_dt)

    return new_drift, new_diffusion, init_fn
