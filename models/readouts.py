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

"""Readout modules."""

from __future__ import annotations

from collections.abc import Sequence
import functools
from typing import Callable, Optional

import einops
from flax import linen as nn
import jax
from jax import numpy as jnp
from kauldron import kd
from kauldron.typing import Float, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member

from scivid.models import base_modules
from scivid.models import dpt as DPT


class UnnormalizePredAndAddInput(nn.Module):
  """Unnormalizes predictions and adds the input.

  Similar to weather's predictors.normalization.InputsAndResiduals.
  Returns last_input_x + inner_readout(y) * residual_std + residual_mean.

  Attributes:
    inner_readout: this module wraps an inner readout module with learned params
    residual_mean: the mean used to unnormalize the residual.
    residual_std: the std used to unnormalize the residual.
  """

  inner_readout: nn.Module
  residual_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
  residual_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

  @nn.compact
  def __call__(self, x: Float['*b t h w c'], y: Float['*b t h w c']):
    res = self.inner_readout(y)
    mean = jnp.full_like(x, fill_value=jnp.array(self.residual_mean))
    std = jnp.full_like(x, fill_value=jnp.array(self.residual_std))
    num_target_frames = res.shape[-4]
    last_x = x[..., -1:, :, :, :]
    last_x = jnp.concatenate([last_x] * num_target_frames, axis=1)
    return last_x + res * std + mean


class DPTReadout(nn.Module):
  """Detokenize to pixel-space patches using a Dense Prediction Transformer.

  The implementation below predicts pred_seq_len output frames, each having
  output_channels channels. Note that the current basic implementation
  only uses the features *from the last frame* to make the predictions.
  """

  output_resolution: tuple[int, int] = (384, 384)
  pred_seq_len: int = 16
  output_channels: int = 3
  dpt_feature_dims: tuple[int, int, int, int] = (256, 256, 256, 256)
  dpt_intermediate_channels: int = 128

  @nn.compact
  def __call__(self, x: Float['b t h w d']) -> Float['b t h w d']:
    input_resolution = x.shape[2:4]

    # Currently, we use only the last frame as input
    x = x[:, -1, :, :, :]

    x = einops.rearrange(
        x, 'b h w d -> b (h w) d'
    )  # flatten spatial dims (required for this DPT implementation)

    # use DPT to predict all of the output frames
    dpt_model = DPT.DensePredictionTransformer(
        input_feature_map_resolution=input_resolution,
        output_resolution=self.output_resolution,
        output_channels=self.pred_seq_len * self.output_channels,
        readout_mapping='',
        non_negative_output=False,
        init_to_zero=True,
        invert_model_output=False,
        use_post_resize_conv_layer=False,
        feature_dims=self.dpt_feature_dims,
        feature_dim_depth_estimation_head=self.dpt_intermediate_channels,
    )

    x = dpt_model(x)

    # reshape the result to [b, pred_seq_len, h, w, output_channels]
    x = einops.rearrange(
        x,
        'b h w (t c) -> b t h w c',
        t=self.pred_seq_len,
        c=self.output_channels,
    )

    return x


class LinearReadout(nn.Module):
  """Linear classifier readout."""

  num_classes: int
  reduce_axis: Sequence[int] = ()
  num_test_clips: int = 1
  is_training = kd.nn.train_property()

  @nn.compact
  def __call__(self, inputs):
    if self.reduce_axis:
      inputs = jnp.mean(inputs, axis=self.reduce_axis)
    out = nn.Dense(self.num_classes, dtype=inputs.dtype)(inputs)

    if self.num_test_clips > 1 and not self.is_training:  # multi-clip eval
      out = nn.softmax(out, axis=-1)
      out = einops.reduce(
          out, '(b n) ...-> b ...', 'mean', n=self.num_test_clips
      )
    return out


class EulerianPersistence(nn.Module):
  """Eulerian persistence baseline.

  Returns the last input frame repeated num_output_frames times.

  Attributes:
    num_output_frames: number of output frames to return.
  """

  num_output_frames: int

  @nn.compact
  def __call__(self, x: Float['*b t h w c']):
    last_x = x[..., -1:, :, :, :]
    repeat_last_x = jnp.concatenate([last_x] * self.num_output_frames, axis=1)
    return repeat_last_x


class AttentionReadout(nn.Module):
  """Cross-attention readout with learnable latent queries."""

  num_classes: int
  num_params: int
  num_heads: int
  num_queries: int = 1
  match_vjepa_implementation: bool = True
  add_temporal_posenc: bool = True
  num_test_clips: int = 1
  is_training = kd.nn.train_property()
  dropout_rate: float = 0.0

  output_shape: tuple[int, int, int] | tuple[int, int, int, int] | None = None
  decoding_patch_size: tuple[int, int, int] | None = None

  def __post_init__(self):
    if (self.output_shape is None and self.decoding_patch_size is not None) or (
        self.output_shape is not None and self.decoding_patch_size is None
    ):
      raise ValueError(
          'Both output_shape and decoding_patch_size must be specified or None.'
      )
    super().__post_init__()

  @nn.compact
  @typechecked
  def __call__(
      self, inputs: Float['B T N C'], queries: Optional[Float['B Q D']] = None
  ):
    num_params_per_head = self.num_params // self.num_heads
    if num_params_per_head * self.num_heads != self.num_params:
      raise ValueError(
          f'num_params ({self.num_params}) must be a multiple of num_heads'
          f' ({self.num_heads}).'
      )
    # Cross-attend from a (different) learned token into each of the given sets.
    feats = inputs

    if self.match_vjepa_implementation:
      # Normalize the input features first
      feats = nn.LayerNorm(dtype=feats.dtype)(feats)
      use_bias = True
    else:
      use_bias = False

    # Optionally add learned posenc to the representation.
    if self.add_temporal_posenc:
      check_type(feats, Float['B T N C'])
      posenc = kd.nn.LearnedEmbedding(name='temporal_posenc')(
          feats.shape, axis=-3
      )
      posenc = posenc.astype(feats.dtype)
      feats += posenc
    feats = einops.rearrange(feats, '... T N C -> ... (T N) C')

    if queries is None:
      # Initialize Learnable queries.
      num_queries = self.num_queries
      query = self.param(
          'query',
          nn.initializers.normal(stddev=0.02),
          [num_queries, self.num_heads, num_params_per_head],
          feats.dtype,
      )
      query = jnp.broadcast_to(query, (feats.shape[0],) + query.shape)
    else:
      num_queries = queries.shape[-2]
      query = nn.Dense(self.num_heads * num_params_per_head, dtype=feats.dtype)(
          queries
      )
      query = einops.rearrange(
          query, '... Q (h n) -> ... Q h n', h=self.num_heads
      )

    # Cross-attention
    key_val_dense = functools.partial(
        nn.DenseGeneral,
        features=(self.num_heads, num_params_per_head),
        axis=-1,
        dtype=feats.dtype,
        use_bias=use_bias,
    )
    key = key_val_dense(name='key_embedding')(feats)
    value = key_val_dense(name='value_embedding')(feats)

    token = nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
    )

    token = nn.Dropout(rate=self.dropout_rate)(
        token, deterministic=not self.is_training
    )
    token = einops.rearrange(token, '...  Q N c -> ... Q (N c)')

    if self.match_vjepa_implementation:
      # Extra MLP layer with residual connection.
      query = einops.rearrange(query, '... Q N c -> ... Q (N c)')

      token = query + nn.Dense(self.num_params, dtype=token.dtype)(token)
      residual = token
      token = nn.LayerNorm(dtype=token.dtype)(token)
      token = base_modules.MLP(
          hidden_size=self.num_params * 4,
          num_hidden_layers=1,
      )(token)
      token = token + residual

    if num_queries == 1:
      # Squeeze the num_queries dimension.
      token = jnp.squeeze(token, axis=-2)

    out = nn.Dense(self.num_classes, dtype=token.dtype)(token)
    if self.output_shape is not None and self.decoding_patch_size is not None:
      channel_dim = self.output_shape[-1] if len(self.output_shape) == 4 else 1
      # Rearrange the output tensor to match the desired output shape, by
      # reshaping the pixels and patches dimensions.
      out = einops.rearrange(
          out,
          'B (n_pixels_patch0 n_pixels_patch1 n_pixels_patch2) (patch_size0'
          ' patch_size1 patch_size2 c) -> B (n_pixels_patch0 patch_size0)'
          ' (n_pixels_patch1 patch_size1) (n_pixels_patch2 patch_size2) c',
          patch_size0=self.decoding_patch_size[0],
          patch_size1=self.decoding_patch_size[1],
          patch_size2=self.decoding_patch_size[2],
          n_pixels_patch0=self.output_shape[0] // self.decoding_patch_size[0],
          n_pixels_patch1=self.output_shape[1] // self.decoding_patch_size[1],
          n_pixels_patch2=self.output_shape[2] // self.decoding_patch_size[2],
          c=channel_dim,
      )

    # note: these options only make sense for classification-type tasks
    if self.num_test_clips > 1 and not self.is_training:  # multi-clip eval
      out = nn.softmax(out, axis=-1)
      out = einops.reduce(
          out,
          '(b n) ...-> b ...',
          'mean',
          n=self.num_test_clips,
      )
    return out


class ShiftPred(nn.Module):
  """Shifts the predictions by provided values."""

  inner_readout: nn.Module

  @nn.compact
  def __call__(self, inputs: Float, output_offsets: Float):
    pred = self.inner_readout(inputs)
    return {'shifted_pred': pred + output_offsets, 'unshifted_pred': pred}


class TrackingReadoutWrapper(nn.Module):
  """Wrapper around cross-attention readout for tracking tasks.

  The __call__ method takes in
    inputs: [B, T, N, C] features
    queries: [B, Q, D] queries
  and returns a dict with the following fields:
    values: [B, T, Q, d]
    logits_visible: None or [B, T, Q, 1] (if self.predict_visibility is True).
    logits_certainty: None or [B, T, Q, 1] (if self.use_certainty is True).
    visible: None or [B, T, Q, 1] (if self.predict_visibility is True).
  """

  attention_readout: AttentionReadout
  query_initializer: nn.Module | None = None
  output_activation: Optional[Callable[[Float['*b']], Float['*b']]] = None
  predict_visibility: bool = False
  use_certainty: bool = True
  certainty_threshold: float = 0.5
  # Number of frames predicted per query. This is used to reshape the output
  # from the attention readout.
  num_frames_per_query: int = 1
  # Number of frames to tile the query. This allows us to repeat the same query
  # multiple times for different frames and uses a learned temporal posenc over
  # time axis.
  temporal_tile_size: int = 1

  @nn.compact
  @typechecked
  def __call__(
      self,
      inputs: Float['B T N C'],
      queries: Float['B Q D'],
  ):
    if self.use_certainty and not self.predict_visibility:
      raise ValueError('Cannot use certainty if visibility is not predicted.')
    if self.temporal_tile_size < 1:
      raise ValueError('Temporal tile size must greater than 0.')

    if self.query_initializer is not None:
      queries = self.query_initializer(queries)

    if self.temporal_tile_size > 1:
      queries = einops.repeat(
          queries,
          'B Q D -> B Q k D',  # k = temporal_tile_size
          k=self.temporal_tile_size,
      )
      posenc = kd.nn.LearnedEmbedding(name='temporal_tile_posenc')(
          queries.shape, axis=-2
      )
      posenc = posenc.astype(queries.dtype)
      queries += posenc
      queries = einops.rearrange(
          queries,
          'B Q k D -> B (Q k) D',
      )

    out = self.attention_readout(inputs, queries)
    # Reshape temporal_tile_size to the last dimension.
    out = einops.rearrange(
        out,
        'B (Q k) C -> B Q (k C)',
        k=self.temporal_tile_size,
    )
    # For each query, we have num_frames_per_query * num_classes outputs.
    out = einops.rearrange(
        out,
        '... Q (k t c) -> ... (k t) Q c',
        k=self.temporal_tile_size,
        t=self.num_frames_per_query,
        c=self.attention_readout.num_classes // self.num_frames_per_query,
    )

    out_values = out
    logits_visible = None
    logits_certainty = None
    visible = None
    if self.predict_visibility:
      logit_size = 1
      if self.use_certainty:
        logit_size += 1
      # Last logit_size values are logits, everything else are values.
      out_values = out[..., :-logit_size]
      out_logits = out[..., -logit_size:]

      # Predict visibility.
      logits_visible = out_logits[..., :1]
      visible = jax.nn.sigmoid(logits_visible)  # [B, T, Q, 1]

      if self.use_certainty:
        # Predict certainty.
        logits_certainty = out_logits[..., 1:]
        certainty = jax.nn.sigmoid(logits_certainty)  # [B, T, Q, 1]
        # Use visibility and certainty to predict visibility.
        visible = (visible * certainty > self.certainty_threshold).astype(
            jnp.float32
        )

    if self.output_activation is not None:
      out_values = self.output_activation(out_values)

    return {
        'values': out_values,  # [B, T, Q, d]
        'logits_visible': logits_visible,  # None or [B, T, Q, 1]
        'logits_certainty': logits_certainty,  # None or [B, T, Q, 1]
        'visible': visible,  # None or [B, T, Q, 1]
    }
