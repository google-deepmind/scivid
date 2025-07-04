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

"""Shared utilities for setting up optimizers."""

from absl import logging
import jax
import jax.numpy as jnp
from kauldron import kontext
import optax


def zero_grads():
  def init_fn(_):
    return ()

  def update_fn(updates, state, params=None):
    del state  # unused
    del params  # unused
    return jax.tree.map(jnp.zeros_like, updates), ()

  return optax.GradientTransformation(init_fn, update_fn)


def make_ignore_base_model_filter_fn():
  """Optax filter to freeze the 'model' module, assumed to be pretrained."""

  def tree_filter_fn(params):
    def filter_fn(path, val):
      del val
      path_str = str(kontext.Path.from_jax_path(path))
      res = path_str.startswith("model")
      if res:
        logging.info("readout wrapper freezing: %s", path_str)
      else:
        logging.info("readout wrapper training: %s", path_str)
      return "frozen" if res else "trained"

    return jax.tree_util.tree_map_with_path(filter_fn, params)

  return tree_filter_fn


def _init_empty_state(params) -> optax.EmptyState:
  """Init function for a :class:`GradientTransformation` with empty state."""
  del params
  return optax.EmptyState()


def scale_backbone(
    alpha: float, readout_prefix: str = "readout"
) -> optax.GradientTransformation:
  """Scale updates by some fixed scalar `alpha`.

  Args:
    alpha: A scalar corresponding to a fixed scaling factor for updates.
    readout_prefix: The prefix of the readout module(s).

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_util.tree_map_with_path(
        lambda path, g: g
        if path[0].key.startswith(readout_prefix)
        else alpha * g,
        updates,
    )

    return updates, state

  return optax.GradientTransformation(_init_empty_state, update_fn)
