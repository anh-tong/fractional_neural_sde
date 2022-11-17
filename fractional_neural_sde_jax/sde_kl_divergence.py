import operator
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax.custom_types import Array, PyTree, Scalar
from diffrax.term import (
    AbstractTerm,
    ControlTerm,
    MultiTerm,
    ODETerm,
    WeaklyDiagonalControlTerm,
)


def _kl_diagonal(drift: Array, diffusion: Array):
    """This is the case where diffusion matrix is
    a diagonal matrix
    """
    diffusion = jnp.where(
        jax.lax.stop_gradient(diffusion) > 1e-7,
        diffusion,
        jnp.full_like(diffusion, fill_value=1e-7) * jnp.sign(diffusion),
    )
    scale = drift / diffusion
    return 0.5 * jnp.sum(scale**2)


def _kl_full_matrix(drift: Array, diffusion: Array):
    """General case"""
    scale = jnp.linalg.pinv(diffusion) @ drift
    return 0.5 * jnp.sum(scale**2)


def _assert_array(x: Any):
    assert isinstance(
        x, jnp.ndarray
    ), "`sde_kl_divergence` can only handle array-value  drifts and diffusions"


def _handle(drift: Array, diffusion: Array):
    """According to the shape of drift and diffusion,
    select the right way to compute KL divergence
    """
    _assert_array(drift)
    _assert_array(diffusion)
    if drift.shape == diffusion.shape:
        return _kl_diagonal(drift, diffusion)
    else:
        return _kl_full_matrix(drift, diffusion)


def _kl_block_diffusion(drift: PyTree, diffusion: PyTree):
    """The case where diffusion matrix is a block diagonal matrix"""
    kl = jtu.tree_map(
        _handle,
        drift,
        diffusion,
    )

    kl = jtu.tree_reduce(
        operator.add,
        kl,
    )
    return kl


class _AugDrift(AbstractTerm):

    drift1: ODETerm
    drift2: ODETerm
    diffusion: AbstractTerm

    def vf(self, t: Scalar, y: PyTree, args) -> PyTree:

        y, _ = y

        # check if there is context
        if len(args) > 0:
            context = args[0]
        else:
            context = None
        aug_y = y if context is None else jnp.concatenate([y, context(t)], axis=-1)

        drift1 = self.drift1.vf(t, aug_y, args[1:])
        drift2 = self.drift2.vf(t, y, args[1:])

        drift = jtu.tree_map(operator.sub, drift1, drift2)
        diffusion = self.diffusion.vf(t, y, args[1:])

        # get tree structure of drift and diffusion
        drift_tree_structure = jtu.tree_structure(drift)
        diffusion_tree_structure = jtu.tree_structure(diffusion)

        if drift_tree_structure == diffusion_tree_structure:
            # drift and diffusion has the same tree structure
            # check the shape to determine how to compute KL
            # however, it does not check the abstract yet

            if isinstance(drift, jnp.ndarray):
                # this case PyTree is (*)

                # here we check the abstract level of ControlTerm
                if isinstance(self.diffusion, WeaklyDiagonalControlTerm):
                    # diffusion must be jnp.ndarrary as well because
                    # diffusion and drift has the same structure
                    # therefore we don't need to check type of diffusion here
                    kl_divergence = _kl_diagonal(drift, diffusion)
                elif isinstance(self.diffusion, ControlTerm):
                    kl_divergence = _kl_full_matrix(drift, diffusion)
            else:
                # a more general case, we assume that on each leave,
                # if drift and diffusion have the same shape
                #   -> WeaklyDiagonalControlTerm
                # else
                #   -> ControlTerm
                kl_divergence = _kl_block_diffusion(drift, diffusion)
        else:
            raise ValueError(
                "drift and diffusion should have the same PyTree structure"
                + f" \n {drift_tree_structure} != {diffusion_tree_structure}"
            )
        return drift1, kl_divergence

    @staticmethod
    def contr(t0: Scalar, t1: Scalar) -> Scalar:
        return t1 - t0

    @staticmethod
    def prod(vf: PyTree, control: Scalar) -> PyTree:
        return jtu.tree_map(lambda v: control * v, vf)


class _AugControlTerm(AbstractTerm):

    control_term: AbstractTerm

    def __init__(self, term: AbstractTerm) -> None:
        self.control_term = term

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        y, _ = y
        args = args[1:]
        vf = self.control_term.vf(t, y, args)
        return vf, 0.0

    def contr(self, t0: Scalar, t1: Scalar) -> PyTree:
        return self.control_term.contr(t0, t1), 0.0

    def vf_prod(self, t: Scalar, y: PyTree, args: PyTree, control: PyTree) -> PyTree:
        y, _ = y
        return self.control_term.vf_prod(t, y, args, control), 0.0

    def prod(self, vf: PyTree, control: PyTree) -> PyTree:
        vf, _ = vf
        control, _ = control
        return self.control_term.prod(vf, control), 0.0


def sde_kl_divergence(
    drift1: ODETerm, drift2: ODETerm, diffusion: AbstractTerm, y0: PyTree
) -> Tuple[MultiTerm, PyTree]:
    aug_y0 = (y0, 0.0)
    aug_drift = _AugDrift(drift1, drift2, diffusion)
    aug_control = _AugControlTerm(diffusion)
    aug_sde = MultiTerm(aug_drift, aug_control)
    return aug_sde, aug_y0
