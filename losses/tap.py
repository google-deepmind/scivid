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

"""Losses for TAPIR."""

import dataclasses

import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.losses import base
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import optax


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Huber(base.Loss):
  """Huber loss for point track prediction."""

  delta: float = 1.0

  pred_points: kontext.Key = kontext.REQUIRED
  target_points: kontext.Key = kontext.REQUIRED
  normalize_by: str = "values"

  @typechecked
  def get_values(
      self,
      pred_points: Float["*a 2"],
      target_points: Float["*a 2"],
  ) -> Float["*a 1"]:
    error = pred_points - target_points
    error = jnp.clip(error, -1e8, 1e8)  # add magnitude bound to prevent nan
    distsqr = jnp.sum(jnp.square(error), axis=-1, keepdims=True)
    dist = jnp.sqrt(distsqr + 1e-12)  # add eps to prevent nan
    loss = jnp.where(
        dist < self.delta,
        distsqr / 2,
        self.delta * (jnp.abs(dist) - self.delta / 2),
    )
    return loss


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Certainty(base.Loss):
  """Loss for point track uncertainty prediction.

  A point prediction is certain if it falls within threshold of ground truth.
  The 3rd term of the loss in Equation (1) of TAPIR paper
  https://arxiv.org/abs/2306.08637
  """

  threshold: float = 1.0

  logits: kontext.Key = kontext.REQUIRED
  pred_points: kontext.Key = kontext.REQUIRED
  target_points: kontext.Key = kontext.REQUIRED
  normalize_by: str = "values"

  @typechecked
  def get_values(
      self,
      logits: Float["*a 1"],
      pred_points: Float["*a 2"],
      target_points: Float["*a 2"],
  ) -> Float["*a 1"]:
    pred_points = jax.lax.stop_gradient(pred_points)
    error = pred_points - target_points
    distsqr = jnp.sum(jnp.square(error), axis=-1, keepdims=True)
    is_certain = (distsqr <= self.threshold**2).astype(logits.dtype)
    loss = optax.sigmoid_binary_cross_entropy(logits, is_certain)
    return loss
