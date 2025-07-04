# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Modules based on DPT https://arxiv.org/abs/2103.13413."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


Array = jnp.ndarray
ArrayTree = Union[
    Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]
]  # pytype: disable=not-supported-yet


class DensePredictionTransformerReassembleBlock(nn.Module):
  """Constructs a reassemble operation used in DensePredictionTransformer.

  A reassemble operation consists of:
    1. Processing the readout token.
    2. Concatenating the tokens into a spatial feature map.
    3. Transforming the feature map via two CNN layers.

  Attributes:
    output_feature_dim: Feature dimensionality of the output.
    strides: Strides used in the final CNN layer.
    input_feature_map_resolution: Resolution of the spatial feature map.
    readout_mapping: How to process the readout token.
    use_transposed_conv: Whether to use a TransposedConv (for upsampling).
    normalize_feature_map: Whether to normalize the ViT feature maps.
  """

  output_feature_dim: int
  strides: tuple[int, int]
  input_feature_map_resolution: tuple[int, int]
  readout_mapping: str
  use_transposed_conv: bool = False
  normalize_feature_map: bool = False

  @typechecked
  @nn.compact
  def __call__(self, tokens: Float["B N D"]) -> Float["*a"]:

    batch_size, num_tokens, input_feature_dim = tokens.shape

    # Optional Layer-norm in case ViT maps are not normalized.
    if self.normalize_feature_map:
      tokens = nn.LayerNorm()(tokens)

    # Read block that handles the read-out token at position 0.
    if not self.readout_mapping:
      pass
    elif self.readout_mapping == "ignore":
      tokens = tokens[:, 1:]
    elif self.readout_mapping == "add":
      readout_token = tokens[:, 0:1]
      tokens = tokens[:, 1:] + readout_token
    elif self.readout_mapping == "project":
      tiled_readout_token = jnp.broadcast_to(
          tokens[:, 0:1], (batch_size, num_tokens - 1, input_feature_dim)
      )
      tokens = jnp.concatenate((tokens[:, 1:], tiled_readout_token), axis=2)
      tokens = nn.Dense(
          features=input_feature_dim,
          use_bias=True,
          name="readout",
          dtype=tokens.dtype,
      )(tokens)
      tokens = nn.gelu(tokens, approximate=True)
    else:
      raise ValueError(f"Unknown readout mapping: {self.readout_mapping}")

    # Concatenate spatially.
    tokens = jnp.reshape(
        tokens,
        (
            batch_size,
            self.input_feature_map_resolution[0],
            self.input_feature_map_resolution[1],
            input_feature_dim,
        ),
    )

    # Resample spatially.
    tokens = nn.Conv(
        name="conv_resample_0",
        features=self.output_feature_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=True,
        padding="SAME",
        dtype=tokens.dtype,
    )(tokens)

    conv_module = nn.ConvTranspose if self.use_transposed_conv else nn.Conv
    tokens = conv_module(
        name="conv_resample_1",
        features=self.output_feature_dim,
        kernel_size=(3, 3),
        strides=self.strides,
        use_bias=True,
        padding="SAME",
        dtype=tokens.dtype,
    )(tokens)

    return tokens


class DensePredictionTransformerFeatureFusionBlock(nn.Module):
  """Constructs a fusion module used in DensePredictionTransformer.

  A fusion module consists of:
    1. An optional residual block to transform the output of ReassembleBlock.
    2. A residual block to transform the fused inputs.
    3. A spatial upsampling via image resize by a factor of 2.

  Attributes:
    feature_dim: Feature dimensionality throughout the block.
    output_feature_dim: Feature dimensionality of the output.
  """

  feature_dim: int
  output_feature_dim: int

  @typechecked
  @nn.compact
  def __call__(self, x1: Float, x2: Optional[Float] = None):
    """Returns fused output of previous fusion block with reassembled tokens.

    Args:
      x1: The output of the previous fusion block. For the first fusion block in
        the stack, x1 takes on the role of the reassembled tokens and x2=None.
      x2: The reassembled tokens at a particular resolution
    """
    x = x1
    if x2 is not None:
      residual = x2
      x2 = nn.relu(x2)
      x2 = nn.Conv(
          self.feature_dim,
          kernel_size=(3, 3),
          strides=(1, 1),
          name="conv_fusion_0_0",
          dtype=x2.dtype,
      )(x2)
      x2 = nn.relu(x2)
      x2 = nn.Conv(
          self.feature_dim,
          kernel_size=(3, 3),
          strides=(1, 1),
          name="conv_fusion_0_1",
          dtype=x2.dtype,
      )(x2)
      x = x + (x2 + residual)

    residual = x
    x = nn.relu(x)
    x = nn.Conv(
        self.feature_dim,
        kernel_size=(3, 3),
        strides=(1, 1),
        name="conv_fusion_1_0",
        dtype=x.dtype,
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        self.feature_dim,
        kernel_size=(3, 3),
        strides=(1, 1),
        name="conv_fusion_1_1",
        dtype=x.dtype,
    )(x)
    x = x + residual

    b, h, w, dim = x.shape
    x = jax.image.resize(x, (b, 2 * h, 2 * w, dim), method="bilinear")

    x = nn.Conv(
        self.output_feature_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        name="conv_fusion_out",
        dtype=x.dtype,
    )(x)

    return x


class DensePredictionTransformer(nn.Module):
  """DensePredictionTransformer proposed in https://arxiv.org/abs/2103.13413.

  NOTE: only the implementation for monocular depth prediction is provided.

  At a high level it consits of:
    1. ReassembleBlocks to process ViT encoder features at multiple resolutions.
    2. FusionBlocks that decode the reassembled features gradually.
    3. A depth prediction head for the monocular depth prediction task.

  By default the model outputs inverted depth.

  Attributes:
    input_feature_map_resolution: Resolution of the spatial feature map.
    feature_dims: Feature dims of the reassemble and corresponding fusion block.
    readout_mapping: Read out mapping to be used during reassembling.
    non_negative_output: Whether to ensure non-negative values are returned.
    invert_model_output: Whether to output depth (as opposed to inverse depth).
    output_scale: Output scale used when using invert_model_output.
    output_shift: Output scale used when using invert_model_output.
    output_resolution: Desired output resolution of the predicted depth.
  """

  # Note: DPT github code suggests slightly different strides.

  # (H // p, W // p) where p is patch size.
  input_feature_map_resolution: tuple[int, int] = (384 // 16, 384 // 16)
  normalize_input_feature_maps: bool = False
  feature_dims: tuple[int, int, int, int] = (256, 256, 256, 256)
  feature_dim_depth_estimation_head: int = 128

  # Toggles using the conv layer after the resize. This head can potentially be
  # detrimental when the output number of channels is high).
  use_post_resize_conv_layer: bool = True

  # Possible readout_mapping values {'', 'ignore', 'add', 'project'}
  readout_mapping: str = "project"

  # Output head params.
  non_negative_output: bool = True
  invert_model_output: bool = False
  output_scale: float = 1.0
  output_shift: float = 0.0
  output_resolution: tuple[int, int] = (384, 384)
  output_channels: int = 4
  init_to_zero: bool = False
  customize_bias_init: nn.initializers.Initializer | None = None

  @typechecked
  @nn.compact
  def __call__(
      self,
      latents: Float["b k N D"] | Float["b N D"],
  ) -> Float["b H W 1"] | Float["b H W d"]:

    # tokens is (B, N, D) --> tile to (B, 4, N, D) to obtain multiple maps.
    if len(latents.shape) == 3:
      latents = jnp.tile(latents[:, jnp.newaxis], reps=[1, 4, 1, 1])
    assert latents.shape[1] == 4, "Only 4 maps are supported."

    # Striding assumes a patch-size of 16x16, resolution 1/4 wrt. img.
    reassemble_4 = DensePredictionTransformerReassembleBlock(
        name="ReassembleBlock4",
        output_feature_dim=self.feature_dims[3],
        strides=(4, 4),
        use_transposed_conv=True,
        input_feature_map_resolution=self.input_feature_map_resolution,
        readout_mapping=self.readout_mapping,
        normalize_feature_map=self.normalize_input_feature_maps,
    )

    # Striding assumes a patch-size of 16x16, resolution 1/8 wrt. img.
    reassemble_8 = DensePredictionTransformerReassembleBlock(
        name="ReassembleBlock8",
        output_feature_dim=self.feature_dims[2],
        strides=(2, 2),
        use_transposed_conv=True,
        input_feature_map_resolution=self.input_feature_map_resolution,
        readout_mapping=self.readout_mapping,
        normalize_feature_map=self.normalize_input_feature_maps,
    )

    # Striding assumes a patch-size of 16x16, resolution 1/16 wrt. img.
    reassemble_16 = DensePredictionTransformerReassembleBlock(
        name="ReassembleBlock16",
        output_feature_dim=self.feature_dims[1],
        strides=(1, 1),
        use_transposed_conv=False,
        input_feature_map_resolution=self.input_feature_map_resolution,
        readout_mapping=self.readout_mapping,
        normalize_feature_map=self.normalize_input_feature_maps,
    )

    # Striding assumes a patch-size of 16x16, resolution 1/32 wrt. img.
    reassemble_32 = DensePredictionTransformerReassembleBlock(
        name="ReassembleBlock32",
        output_feature_dim=self.feature_dims[0],
        strides=(2, 2),
        use_transposed_conv=False,
        input_feature_map_resolution=self.input_feature_map_resolution,
        readout_mapping=self.readout_mapping,
        normalize_feature_map=self.normalize_input_feature_maps,
    )

    # Upsampling in fusion block assumes scale factor of 2 between reassembles.
    encoder_features = reassemble_32(latents[:, 3])
    outputs = DensePredictionTransformerFeatureFusionBlock(
        feature_dim=self.feature_dims[0],
        output_feature_dim=self.feature_dims[1],
        name="FusionBlock32",
    )(encoder_features)

    encoder_features = reassemble_16(latents[:, 2])
    outputs = DensePredictionTransformerFeatureFusionBlock(
        feature_dim=self.feature_dims[1],
        output_feature_dim=self.feature_dims[2],
        name="FusionBlock16",
    )(outputs, encoder_features)

    encoder_features = reassemble_8(latents[:, 1])
    outputs = DensePredictionTransformerFeatureFusionBlock(
        feature_dim=self.feature_dims[2],
        output_feature_dim=self.feature_dims[3],
        name="FusionBlock8",
    )(outputs, encoder_features)

    encoder_features = reassemble_4(latents[:, 0])
    outputs = DensePredictionTransformerFeatureFusionBlock(
        feature_dim=self.feature_dims[3],
        output_feature_dim=self.feature_dims[3],
        name="FusionBlock4",
    )(outputs, encoder_features)

    # Monocular Depth Estimation Head.
    outputs = nn.Conv(
        features=self.feature_dim_depth_estimation_head,
        kernel_size=(3, 3),
        strides=(1, 1),
        name="conv_head_0",
        dtype=outputs.dtype,
    )(outputs)

    # Resize to desired output resolution.
    b, _, _, dim = outputs.shape
    h, w = self.output_resolution
    outputs = jax.image.resize(outputs, (b, h, w, dim), method="bilinear")
    # Apply a conv layer after the resize. Note that this layer might limit the
    # network capacity when the output number of channels is high by introducing
    # an intermediate bottleneck on the number of channels. It might be better
    # to avoid using it if output_channels is larger than 32.
    if self.use_post_resize_conv_layer:
      outputs = nn.Conv(
          features=32,
          kernel_size=(3, 3),
          strides=(1, 1),
          name="conv_head_1",
          dtype=outputs.dtype,
      )(outputs)
      outputs = nn.relu(outputs)
    kwargs = {}
    if self.init_to_zero:
      kwargs["kernel_init"] = nn.initializers.zeros_init()
      kwargs["bias_init"] = nn.initializers.zeros_init()

    if self.customize_bias_init is not None:
      kwargs["bias_init"] = self.customize_bias_init

    outputs = nn.Conv(
        features=self.output_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        name="conv_head_2",
        dtype=outputs.dtype,
        **kwargs,
    )(outputs)
    outputs = nn.relu(outputs) if self.non_negative_output else outputs
    # By default the model produces inverted depth.
    inverse_depth = outputs
    if self.invert_model_output:
      depth = self.output_scale * inverse_depth + self.output_shift
      depth = jnp.clip(depth, min=1e-8)
      depth = 1.0 / depth
      return depth

    return inverse_depth


