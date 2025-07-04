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

"""Miscellaneous base modules."""

from __future__ import annotations

from typing import Optional

from flax import linen as nn
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class MLP(nn.Module):
  """A simple MLP."""

  hidden_size: int
  output_size: Optional[int] = None
  num_hidden_layers: int = 1

  @typechecked
  @nn.compact
  def __call__(self, inputs: Float['*B c']) -> Float['*B d']:
    output_size = self.output_size or inputs.shape[-1]

    x = inputs
    for _ in range(self.num_hidden_layers):
      x = nn.Dense(self.hidden_size, dtype=x.dtype)(x)
      x = nn.gelu(x)
    x = nn.Dense(output_size, dtype=x.dtype)(x)

    return x
