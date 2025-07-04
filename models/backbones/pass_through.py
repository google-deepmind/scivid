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

"""Pass-through model (possibly with downsampling)."""

from typing import Optional

import einshape
import flax.linen as nn
import jax.numpy as jnp
from kauldron.typing import Float  # pylint: disable=g-multiple-import,g-importing-member


class PassThroughModel(nn.Module):
  """A pass-through model on spatiotemporal inputs, possibly with downsampling."""

  patch_size: Optional[tuple[int, int, int]] = None

  @nn.compact
  def __call__(
      self,
      image: Float["*b t h w c"],
  ) -> dict[str, Float["*b t _h _w _c"]]:
    if self.patch_size is not None:
      l, m, n = self.patch_size
      image = einshape.jax_einshape(
          "...(tl)(hm)(wn)c->...thw(lmn)c",
          image,
          l=l,
          m=m,
          n=n,
      )
      image = jnp.mean(image, axis=-2)
    flat_image = einshape.jax_einshape("...thwc->...t(hw)c", image)
    mean_pixel = jnp.mean(flat_image, axis=-2, keepdims=True)
    return {"image": image, "flat_image": flat_image, "mean_pixel": mean_pixel}
