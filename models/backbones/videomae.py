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

"""Wrapper for HuggingFace VideoMAE model."""

import dataclasses
import einops
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from kauldron.typing import Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import torch
from transformers import image_processing_base
from transformers.models.videomae import modeling_videomae


@dataclasses.dataclass(kw_only=True)
class HfModelWrapper(nn.Module, kw_only=True):
  """HuggingFace ViT encoder used in VideoMAE.

  Attributes:
    model: HuggingFace VideoMAEModel used as backbone.
    model_output_dims: Tuple of integers representing the expected feature
      dimensions based on the model outputs.
    torch_device: Torch device to use for the backbone model. 'cpu' or 'cuda'.
    jax_device: JAX device to use for the readout model. 'cpu' or 'gpu'.
  """

  model: modeling_videomae.VideoMAEModel
  model_output_dims: tuple[int, int, int, int] = (8, 14, 14, 768)
  torch_device: torch.device = torch.device("cpu")
  jax_device: str = "cpu"

  def setup(self):
    """This is called once when the module is initialized.

    It determines the device for PyTorch and moves the model to that device.
    """
    self.model.to(self.torch_device)
    self.model.eval()

  def _run_torch_model(self, inputs: Float["*b t h w c"]):
    """This function runs on the GPU, if available, otherwise on CPU."""
    torch_model_inputs = image_processing_base.BatchFeature({
        "pixel_values": (
            torch.from_numpy(np.asarray(inputs)).to(self.torch_device)
        )
    })
    with torch.inference_mode():
      last_hidden_state = self.model(**torch_model_inputs).last_hidden_state
      last_hidden_state_np = last_hidden_state.detach().cpu().numpy()
      return {"last_hidden_state": last_hidden_state_np}

  @nn.compact
  def __call__(
      self,
      image: Float["*b t h w c"],
  ) -> dict[str, Float["*b t _h _w _c"]]:

    b = image.shape[0]
    if len(self.model_output_dims) != 4:
      raise ValueError(
          "Expected model_output_dims to have length 4, but got"
          f" {len(self.model_output_dims)}"
      )
    # Compute feature sizes.
    t, h, w, d, *_ = self.model_output_dims

    expected_output_structure = {
        "last_hidden_state": jax.ShapeDtypeStruct(
            shape=(b, t*h*w, d), dtype=jnp.float32
        ),
    }

    # Encode the video.
    image = einops.rearrange(image, "b t h w c -> b t c h w")
    model_outputs = jax.pure_callback(
        self._run_torch_model,
        expected_output_structure,
        lax.stop_gradient(image),
    )
    model_outputs = jax.device_put(
        model_outputs, jax.devices(self.jax_device)[0]
    )

    features = einops.rearrange(
        model_outputs["last_hidden_state"],
        "b (t h w) d -> b t (h w) d",
        h=h,
        w=w,
        t=t,
    )

    # Upsample along the time dimension to undo temporal downsampling.
    features = jax.image.resize(
        features, (b, image.shape[1], h*w, d),
        "bilinear",
    )

    grid_features = einops.rearrange(
        features,
        "b t (h w) d -> b t h w d",
        h=h,
        w=w,
    )
    return {"grid_features": grid_features, "features": features}
