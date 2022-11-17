import math
from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int


def _zero_param(module: eqx.Module):
    """Make weight and bias to zero"""
    module = eqx.tree_at(
        lambda l: l.weight,
        module,
        jnp.zeros_like(module.weight),
    )
    module = eqx.tree_at(
        lambda l: l.bias,
        module,
        jnp.zeros_like(module.bias),
    )

    return module


class Upsample2D(eqx.Module):

    use_conv: bool
    conv: eqx.nn.Conv2d

    def __init__(
        self,
        channels: int,
        use_conv: Optional[bool] = True,
        out_channels: Optional[int] = None,
        padding: Optional[int] = 1,
        *,
        key: jrandom.PRNGKey,
    ) -> None:
        self.use_conv = use_conv
        self.conv = eqx.nn.Conv2d(
            in_channels=channels,
            out_channels=out_channels if out_channels else channels,
            kernel_size=3,
            padding=padding,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "channel height width"],
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ):

        c, h, w = x.shape
        x = jax.image.resize(
            x,
            shape=(c, h * 2, w * 2),
            method="nearest",
        )
        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample2D(eqx.Module):

    conv: eqx.nn.Conv2d

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: Optional[int] = 1,
        *,
        key: jrandom.PRNGKey,
    ) -> None:
        self.conv = eqx.nn.Conv2d(
            in_channels=channels,
            out_channels=out_channels if out_channels else channels,
            kernel_size=3,
            stride=2,
            padding=padding,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "channel height width"],
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ):
        return self.conv(x)


class ResBlock(eqx.Module):

    # the number of input channels
    channels: int
    # the number of embeding channels
    embedding_channels: int
    # the rate of dropout
    dropout: float
    # number of output channels
    out_channels: int
    use_conv: bool
    use_scale_shift_norm: bool

    # input layer modules
    in_norm: eqx.nn.GroupNorm
    in_conv: eqx.Module

    # embedding layers
    embedding_layer: eqx.Module

    # output layers
    out_norm: eqx.nn.GroupNorm
    out_dropout: eqx.nn.Dropout
    out_conv: eqx.Module

    # skip connection
    skip_connections: eqx.Module

    def __init__(
        self,
        channels: int,
        embedding_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: Optional[bool] = True,
        use_scale_shift_norm=False,
        norm_groups: Optional[int] = 32,
        *,
        key,
    ) -> None:

        (in_key, emb_key, out_key, skip_key) = jrandom.split(
            key,
            4,
        )

        self.channels = channels
        self.embedding_channels = embedding_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        # IN LAYERS : group normalization -> SiLU -> Conv
        self.in_norm = eqx.nn.GroupNorm(groups=norm_groups, channels=channels)
        self.in_conv = eqx.nn.Conv2d(
            in_channels=channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            key=in_key,
        )

        # EMBEDDING LAYER: SiLU -> Linear
        if use_scale_shift_norm:
            out_dim = 2 * self.out_channels
        else:
            out_dim = self.out_channels
        self.embedding_layer = eqx.nn.Linear(
            in_features=self.embedding_channels,
            out_features=out_dim,
            key=emb_key,
        )

        # OUT LAYERS: Group norm-> SiLU -> Dropout -> Conv with Zeros
        self.out_norm = eqx.nn.GroupNorm(groups=norm_groups, channels=self.out_channels)
        self.out_dropout = eqx.nn.Dropout(p=dropout)
        out_conv = eqx.nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            key=out_key,
        )
        self.out_conv = _zero_param(out_conv)

        if self.out_channels == channels:
            self.skip_connections = eqx.nn.Identity()
        elif use_conv:
            self.skip_connections = eqx.nn.Conv2d(
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
                key=skip_key,
            )
        else:
            self.skip_connections = eqx.nn.Conv2d(
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=1,
                key=skip_key,
            )

    def __call__(
        self,
        x: Float[Array, "c h w"],
        embedding: Float[Array, "embedding_dim"],  # noqa: F821
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ):

        h = self.in_norm(x)
        h = jax.nn.silu(h)
        h = self.in_conv(h)

        embedding_out = jax.nn.silu(embedding)
        embedding_out = self.embedding_layer(embedding_out)
        # make sure the same shape
        embedding_out = jnp.reshape(
            embedding_out, embedding_out.shape + (1,) * (h.ndim - 1)
        )
        if self.use_scale_shift_norm:
            scale, shift = jnp.split(embedding_out, 2, axis=0)
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h = h + embedding_out
            h = self.out_norm(h)

        h = jax.nn.silu(h)
        h = self.out_dropout(h, key=key)
        h = self.out_conv(h)

        return self.skip_connections(x) + h


class QKVAttention(eqx.Module):
    """
    This attention module will be used in U-Net
    Query, Key, Value will be projected based on Conv1D
    """

    channels: int
    num_heads: int
    qkv: eqx.nn.Conv1d

    def __init__(self, channels: int, num_heads: int, *, key) -> None:
        self.channels = channels
        self.num_heads = num_heads
        self.qkv = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 3,
            kernel_size=1,
            key=key,
        )

    def __call__(self, x):

        qkv = self.qkv(x)  # (3 * H * C, T)

        q, k, v = jnp.split(qkv, indices_or_sections=3, axis=0)
        ret = eqx.nn.attention.dot_product_attention(q, k, v)
        ret = jnp.reshape(ret, (-1, qkv.shape[-1]))
        return ret


class AttentionBlock(eqx.Module):

    norm: eqx.nn.GroupNorm
    qkv: QKVAttention
    proj_out: eqx.Module

    def __init__(
        self,
        channels: int,
        num_heads: Optional[int] = 1,
        num_head_channels: Optional[int] = -1,
        norm_groups=32,
        *,
        key: jrandom.PRNGKey,
    ) -> None:

        qkv_key, proj_key = jrandom.split(key)

        if num_head_channels == -1:
            num_heads = num_heads
        else:
            num_heads = channels // num_head_channels

        self.norm = eqx.nn.GroupNorm(
            groups=norm_groups,
            channels=channels,
        )

        self.qkv = QKVAttention(
            channels=channels,
            num_heads=num_heads,
            key=qkv_key,
        )

        proj_out = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            key=proj_key,
        )
        self.proj_out = _zero_param(proj_out)

    def __call__(
        self, x: Float[Array, "c h w"], *, key: Optional[jrandom.PRNGKey] = None
    ):
        """
        Architecture:
            - Group Normalization (32, channels)
            - Attention mechanism
            - Conv1d as projector
            - Finally, skip connection from output to input
        """

        c, *spatial = x.shape
        x = jnp.reshape(x, (c, -1))
        x = self.norm(x)
        h = self.qkv(x)
        h = self.proj_out(h)

        # there is a skip connection in attention model
        return (x + h).reshape(c, *spatial)


class AttnDownBlock2D(eqx.Module):

    add_downsample: bool

    resnets: List[ResBlock]
    attentions: List[AttentionBlock]
    downsamplers_0: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        dropout: Optional[float] = 0.0,
        num_layers: Optional[int] = 1,
        attn_num_head_channels: Optional[int] = 1,
        add_downsample: Optional[bool] = True,
        norm_groups: Optional[int] = 32,
        *,
        key=jrandom.PRNGKey,
    ) -> None:

        resnets, attentions = [], []

        for i in range(num_layers):

            input_channels = in_channels if i == 0 else out_channels
            resnet = ResBlock(
                channels=input_channels,
                out_channels=out_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                norm_groups=norm_groups,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            resnets.append(resnet)

            attn = AttentionBlock(
                channels=out_channels,
                num_head_channels=attn_num_head_channels,
                norm_groups=norm_groups,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            attentions.append(attn)

        self.resnets = resnets
        self.attentions = attentions

        self.add_downsample = add_downsample
        if add_downsample:
            self.downsamplers_0 = Downsample2D(input_channels, key=key)
        else:
            self.downsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        hidden_states: Float[Array, "in_channel height width"],
        embedding: Float[Array, "..."],
        *,
        key: jrandom.PRNGKey = None,
    ) -> Tuple[Float[Array, "..."], List[Float[Array, "..."]]]:

        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, embedding, key=key)
            hidden_states = attn(hidden_states, key=key)
            output_states += [hidden_states]

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += [hidden_states]

        return hidden_states, output_states


class AttnUpBlock2D(eqx.Module):

    in_channels: int
    out_channels: int
    embedding_channels: int
    prev_out_channels: int
    add_upsample: bool

    resnets: List[eqx.Module]
    attentions: List[eqx.Module]
    upsamplers_0: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        embedding_channels: int,
        num_layers: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
        attn_num_head_channels: Optional[int] = 1,
        add_upsample: Optional[bool] = True,
        norm_groups: Optional[int] = 32,
        *,
        key: jrandom.PRNGKey,
    ) -> None:

        resnets = []
        attentions = []

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_channels = embedding_channels
        self.prev_out_channels = prev_out_channels
        self.add_upsample = add_upsample

        resnets = []

        for i in range(num_layers):
            res_skip_channels = (
                self.in_channels if (i == num_layers - 1) else self.out_channels
            )
            resnet_in_channels = self.prev_out_channels if i == 0 else self.out_channels

            res_block = ResBlock(
                channels=resnet_in_channels + res_skip_channels,
                embedding_channels=self.embedding_channels,
                out_channels=self.out_channels,
                dropout=dropout,
                norm_groups=norm_groups,
                key=key,
            )

            key = jrandom.split(key, 1)[0]
            resnets.append(res_block)

            attn = AttentionBlock(
                channels=out_channels,
                num_head_channels=attn_num_head_channels,
                norm_groups=norm_groups,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            attentions.append(attn)

        self.resnets = resnets
        self.attentions = attentions

        self.add_upsample = add_upsample
        if add_upsample:
            self.upsamplers_0 = Upsample2D(out_channels, key=key)
        else:
            self.upsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        hidden_states: Float[Array, "channel height width"],
        res_hidden_states_list: List[Float[Array, "..."]],
        embedding: Float[Array, "..."],
        *,
        key: jrandom.PRNGKey,
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_list.pop()
            hidden_states = jnp.concatenate([hidden_states, res_hidden_states], axis=0)
            hidden_states = resnet(hidden_states, embedding, key=key)
            hidden_states = attn(hidden_states, key=key)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class MidBlock2D(eqx.Module):

    resnets: List[eqx.Module]
    attentions: List[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        embedding_channels: int,
        dropout: Optional[float] = 0.0,
        num_layers: Optional[int] = 1,
        attn_num_head_channels: Optional[int] = 1,
        norm_groups: Optional[int] = 32,
        *,
        key: jrandom.PRNGKey,
    ) -> None:

        resnets = [
            ResBlock(
                channels=in_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                norm_groups=norm_groups,
                key=key,
            )
        ]
        key = jrandom.split(key, 1)[0]

        attentions = []
        for _ in range(num_layers):
            attn = AttentionBlock(
                channels=in_channels,
                num_head_channels=attn_num_head_channels,
                norm_groups=norm_groups,
                key=key,
            )
            attentions.append(attn)
            key = jrandom.split(key, 1)[0]

            resnet = ResBlock(
                channels=in_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                norm_groups=norm_groups,
                key=key,
            )
            resnets.append(resnet)
            key = jrandom.split(key, 1)[0]

        self.attentions = attentions
        self.resnets = resnets

    def __call__(
        self,
        x: Float[Array, "channel height width"],
        embedding: Float[Array, "embedding_dim"],  # noqa: F821
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ):
        x = self.resnets[0](x, embedding, key=key)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x, key=key)
            x = resnet(x, embedding, key=key)
        return x


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    max_period=1e4,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
):
    """
    create time proposition encoding
    Args:
        timesteps: float or ndim=1
        dim: output dimension
    """
    half = embedding_dim // 2
    freqs = jnp.arange(start=0, stop=half) / (half - downscale_freq_shift)
    freqs = -jnp.exp(-math.log(max_period) * freqs)
    args = timesteps * freqs

    args = args * scale

    if flip_sin_to_cos:
        embedding = jnp.concatenate(
            [jnp.sin(args), jnp.cos(args)],
            axis=-1,
        )
    else:
        embedding = jnp.concatenate(
            [jnp.cos(args), jnp.sin(args)],
            axis=-1,
        )
    if embedding_dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.array([0.0])],
            axis=-1,
        )
    return embedding


class UNet2DModel(eqx.Module):

    # pre-process
    conv_in: eqx.Module

    # UNet core architecture
    down_blocks: List[eqx.Module]
    mid_block: eqx.Module
    up_blocks: List[eqx.Module]

    # post-process
    norm_out: eqx.nn.GroupNorm
    activation_out: eqx.Module
    conv_out: eqx.Module

    # other params
    time_embedding_dim: int
    freq_shift: float
    max_period: float

    def __init__(
        self,
        # network architecture
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int] = (32, 64),
        layers_per_block: int = 2,
        attn_head_dim: int = 8,
        norm_groups: int = 32,
        # time embedding
        time_embedding_dim=64,
        freq_shift: float = 1.0,
        max_period: float = 1e4,
        *,
        key: jrandom.PRNGKey,
    ) -> None:
        """
        Unconditional UNet for 2D data (i.e., images)
        Args:
            in_channels (int, optional): the number of channels in input. Defaults to 3.
            out_channels (int, optional): the number of channels in ouput. Defaults to 3.
            block_out_channels (Tuple[int], optional): the number of channels of output in each block.
                Defaults to (224, 448).
            layers_per_block (int, optional): the number of ResNet layers in each block. Defaults to 2.
            attn_head_dim (int, optional): the dimension of attention heads when attention is included. Defaults to 8.
            freq_shift (int, optional): shift of positional encoding. Defaults to 0.
            key (jrandom.PRNGKey): JAX random generator key
        """

        # pre-process
        self.conv_in = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            padding=1,
            key=key,
        )
        key = jrandom.split(key, 1)[0]

        # time embbeding
        self.time_embedding_dim = time_embedding_dim
        self.freq_shift = freq_shift
        self.max_period = max_period

        down_blocks = []
        up_blocks = []

        # down blocks
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = AttnDownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                embedding_channels=time_embedding_dim,
                num_layers=layers_per_block,
                add_downsample=not is_final_block,
                attn_num_head_channels=attn_head_dim,
                norm_groups=norm_groups,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            down_blocks.append(down_block)

        # middle block
        mid_block = MidBlock2D(
            in_channels=block_out_channels[-1],
            embedding_channels=time_embedding_dim,
            attn_num_head_channels=attn_head_dim,
            norm_groups=norm_groups,
            key=key,
        )
        key = jrandom.split(key, 1)[0]

        # up block
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_out_channels = output_channel
            output_channel = reversed_block_out_channels[i]
            # input channel is either the next one or the last channel
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            is_final_block = i == len(block_out_channels) - 1
            up_block = AttnUpBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_out_channels=prev_out_channels,
                embedding_channels=time_embedding_dim,
                num_layers=layers_per_block + 1,  # having extra layer
                add_upsample=not is_final_block,
                attn_num_head_channels=attn_head_dim,
                norm_groups=norm_groups,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            up_blocks.append(up_block)

        self.down_blocks = down_blocks
        self.mid_block = mid_block
        self.up_blocks = up_blocks

        # post-process
        self.norm_out = eqx.nn.GroupNorm(
            groups=norm_groups, channels=block_out_channels[0]
        )
        self.activation_out = eqx.nn.Lambda(jax.nn.silu)
        self.conv_out = eqx.nn.Conv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "channel height weight"],
        timestep: Union[Float, Int],
        *,
        key: jrandom.PRNGKey = None,
    ) -> Float[Array, "channel height weight"]:

        time_embedding = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.time_embedding_dim,
            max_period=self.max_period,
            downscale_freq_shift=self.freq_shift,
        )

        # pre-process
        x = self.conv_in(x)

        # down
        down_block_residuals = [x]
        for down_block in self.down_blocks:
            x, residual_x = down_block(x, time_embedding, key=key)
            down_block_residuals += residual_x

        # middle
        x = self.mid_block(x, time_embedding, key=key)

        # up
        for up_block in self.up_blocks:
            residual_x = down_block_residuals[-len(up_block.resnets) :]
            down_block_residuals = down_block_residuals[: -len(up_block.resnets)]

            x = up_block(x, residual_x, time_embedding, key=key)

        # post-process
        x = self.norm_out(x)
        x = self.activation_out(x)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":

    import random

    jax.config.update("jax_platform_name", "cpu")

    def test_attn_downblock2d(get_key):

        in_channels = 32
        out_channels = 32
        embedding_channels = 10

        down_block = AttnDownBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_channels=embedding_channels,
            dropout=0.5,
            num_layers=2,
            add_downsample=True,
            norm_groups=16,
            key=get_key(),
        )

        input = jrandom.normal(key=get_key(), shape=(in_channels, 10, 10))
        temb = jrandom.normal(key=get_key(), shape=(embedding_channels,))
        down_block(input, temb, key=get_key())

    def test_attn_upblock2d(get_key):

        in_channels = 32
        out_channels = 32
        prev_out_channels = 32
        embedding_channels = 10

        up_block = AttnUpBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_out_channels=prev_out_channels,
            embedding_channels=embedding_channels,
            dropout=0.5,
            num_layers=2,
            norm_groups=16,
            key=get_key(),
        )

        hidden_states = jrandom.normal(key=get_key(), shape=(in_channels, 10, 10))
        prev = [jrandom.normal(key=get_key(), shape=(prev_out_channels, 10, 10))] * 2
        temb = jrandom.normal(key=get_key(), shape=(embedding_channels,))
        up_block(hidden_states, prev, temb, key=get_key())

    def test_unet2d(get_key):

        in_channels = 3
        out_channels = 3
        image_size = 28

        unet = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=1,
            block_out_channels=(8, 8, 16),
            norm_groups=8,
            key=get_key(),
        )

        input = jrandom.normal(
            key=get_key(), shape=(in_channels, image_size, image_size)
        )
        timestep = jnp.array(10.0)
        output = unet(timestep, input)

        assert output.shape == (out_channels, image_size, image_size)

    get_key = lambda: jrandom.PRNGKey(random.randint(1, 2e10))

    test_attn_downblock2d(get_key)
    test_attn_upblock2d(get_key)
    test_unet2d(get_key)
