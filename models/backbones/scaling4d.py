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

"""Model and checkpoint loader for Scaling4D model.

paper: https://arxiv.org/abs/2412.15212
github repo: https://github.com/google-deepmind/representations4d
"""

import dataclasses
from typing import TypeVar

import chex
import einops
import flax
from flax import linen as nn
import jax.numpy as jnp
from kauldron import checkpoints
from kauldron.modules import pos_embeddings
from kauldron.modules import vit as kd_vit
from kauldron.typing import Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


from representations4d.models import model as model_lib
from representations4d.utils import checkpoint_utils


_T = TypeVar("_T")


def npload(fname: str) -> dict[str, np.ndarray]:
  with open(fname, "rb") as f:
    loaded = np.load(f, allow_pickle=False)
    return dict(loaded)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Loader(checkpoints.AbstractPartialLoader):
  """Loader for pretrained weights from a .npz file.

  This loader is designed to load pretrained weights for a model backbone
  from a .npz file. The weights are expected to be in a flat dictionary format,
  which is then reconstructed into a nested parameter tree.

  Attributes:
    checkpoint_path: Path to the .npz file containing the pretrained weights.
    backbone_prefix: The key in the state `params` where the backbone weights
      should be loaded.
  """

  checkpoint_path: str
  backbone_prefix: str

  def transform(self, state: _T) -> _T:
    """Transforms the state by updating it with pre-trained weights."""
    # Load weights from .npz file.
    flat_state = npload(self.checkpoint_path)

    params = flax.core.unfreeze(state.params)  # pylint: disable=attribute-error

    restored_model_params = checkpoint_utils.recover_tree(flat_state)

    # Add additional "encoder" level to the restored params under which
    # "encoder" and "processor" are nested.
    encoder_params = restored_model_params["params"].pop("encoder")
    processor_params = restored_model_params["params"].pop("processor")
    restored_model_params["params"]["encoder"] = {
        "encoder": encoder_params,
        "processor": processor_params,
    }

    # Ensure that restored param shapes match backbone model shapes.
    chex.assert_trees_all_equal_shapes(
        params[self.backbone_prefix], restored_model_params["params"]
    )
    params[self.backbone_prefix] = restored_model_params["params"]
    return state.replace(params=flax.core.freeze(params))  # pylint: disable=attribute-error


class Scaling4DModel(nn.Module):
  """Scaling4D model.

  Attributes:
    model_size: Size of the ViT model, e.g. "B".
    model_patch_size: Tuple of (temporal, height, width) patch size.
    image_size: Tuple of (height, width) of input image.
    num_input_frames: Number of input frames.
    readout_depth: Depth of the readout from the transformer layers.
    dtype: Data type, eg. jnp.float32.
  """

  model_size: str = "B"
  model_patch_size: tuple[int, int, int] = (2, 16, 16)
  image_size: tuple[int, int] = (224, 224)
  num_input_frames: int = 16
  readout_depth: float = 0.95
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    """Initialises the model."""
    self.encoder = model_lib.Model(
        encoder=model_lib.Tokenizer(
            patch_embedding=model_lib.PatchEmbedding(
                patch_size=self.model_patch_size,
                num_features=kd_vit.VIT_SIZES[self.model_size][0],
            ),
            posenc=pos_embeddings.LearnedEmbedding(dtype=self.dtype),
            posenc_axes=(-4, -3, -2),
        ),
        processor=model_lib.GeneralizedTransformer.from_variant_str(
            variant_str=self.model_size,
            dtype=self.dtype,
        ),
    )

    self.encoder_to_readout = model_lib.EncoderToReadout(
        embedding_shape=(
            self.num_input_frames // self.model_patch_size[0],
            self.image_size[0] // self.model_patch_size[1],
            self.image_size[1] // self.model_patch_size[2],
        ),
        readout_depth=self.readout_depth,
        num_input_frames=self.num_input_frames,
    )

  @nn.compact
  def __call__(
      self,
      image: Float["*b t h w c"],
  ) -> dict[str, Float["..."]]:
    """Forward pass."""
    all_features = self.encoder(image)
    features = self.encoder_to_readout(all_features)

    h_patch = self.image_size[0] // self.model_patch_size[1]
    w_patch = self.image_size[1] // self.model_patch_size[2]

    grid_features = einops.rearrange(
        features,
        "b t (h w) d -> b t h w d",
        h=h_patch,
        w=w_patch,
    )

    return {"grid_features": grid_features, "features": features}
