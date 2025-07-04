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

"""Basic stats."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import flax.struct
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Count(base.Metric):
  """Count number of samples (total and possibly per-class).

  Attributes:
    labels: Ground truth labels.
    class_names: class names ordered by class index - if provided, per-class
      counts are added.
  """

  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.labels"
  class_names: Sequence[str] = ()

  @flax.struct.dataclass
  class State(base_state.CollectingState):
    """Count state."""

    labels: Int["*b n"]

    @typechecked
    def compute(self) -> dict[str, float]:
      out = super().compute()
      labels = out.labels
      results = {}

      # Add the total number of samples.
      results["total"] = float(labels.shape[0])

      # If class labels are provided, gather per-class information.
      if self.parent.class_names:  # pytype: disable=attribute-error
        # Count number of samples for each label.
        label_counts = labels.sum(0)

        # Populate results with per-class information.
        for class_name, label_count in zip(
            self.parent.class_names,  # pytype: disable=attribute-error
            label_counts,
            strict=True,
        ):
          results[class_name] = float(label_count)

      return results

  @typechecked
  def get_state(
      self,
      labels: Int["*b n"],
  ) -> Count.State:
    return self.State(
        labels=labels,
    )

  def __metric_names__(self) -> list[str]:
    # Add aggregated metric names.
    metric_names = ["total"]

    # Add per-class metric name if class names are provided.
    if self.class_names:
      for class_name in self.class_names:
        metric_names.append(class_name)
    return metric_names
