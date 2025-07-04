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

"""Mean average precision."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import flax.struct
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import sklearn.metrics


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MeanAveragePrecision(base.Metric):
  """Mean average precision metric for multi-label classification.

  Attributes:
    predictions: Predictions for each class.
    labels: Ground truth labels for each class.
    class_names: Names of the classes - if provided, per-class metrics are added
      (including any skipped classes).
    skip_classes: If non-empty, compute a skipped_{classes}_AP metric averaged
      over all but these classes.
    skip_background: Whether to skip the background class for all aggregated
      metrics: mean_AP, control_AP (and if computed, mean_AP_when_skipping_
      classes:{...}), count etc - standard evaluation setting for certain
        datasets.
    background_class_index: Index of the background class.
  """

  predictions: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"
  class_names: Sequence[str] = ()
  skip_classes: Sequence[int] = ()
  skip_background: bool = False
  background_class_index: int = -1

  # pytype: disable=attribute-error
  @flax.struct.dataclass
  class State(base_state.CollectingState):
    """MeanAveragePrecision state."""

    predictions: Float["*b n"]
    labels: Int["*b n"]

    @typechecked
    def compute(self) -> dict[str, float]:
      out = super().compute()
      labels = out.labels
      predictions = out.predictions
      results = {}

      # Compute average precision for each separate class.
      # TODO(b/406164638): require sklearn version this is tested for? (1.6.1)
      per_class_ap = sklearn.metrics.average_precision_score(
          y_true=labels,
          y_score=predictions,
          average=None,
      )

      # If class labels are provided, gather per-class information.
      if self.parent.class_names:
        # Populate results with per-class information.
        for class_name, ap_score in zip(
            self.parent.class_names,
            per_class_ap,
            strict=True,
        ):
          results[f"{class_name}_AP"] = float(ap_score)

      # Maybe skip background class for all aggregated results.
      if self.parent.skip_background:
        labels, keep_mask = self.parent.remove_skipped_classes(
            labels, [self.parent.background_class_index]
        )
        predictions = predictions[:, keep_mask]
        per_class_ap = per_class_ap[keep_mask]

      mean_ap = np.mean(per_class_ap)
      results["mean_AP"] = float(mean_ap)

      # Maybe also compute AP averaged over selected classes; for instance
      # skipping background and/or underrepresented classes to reduce noise.
      if self.parent.skip_classes:
        kept_labels, keep_mask = self.parent.remove_skipped_classes(
            labels, self.parent.skip_classes
        )
        kept_predictions = predictions[:, keep_mask]
        kept_mean_ap = sklearn.metrics.average_precision_score(
            y_true=kept_labels,
            y_score=kept_predictions,
        )
        results[self.parent.skip_classes_mean_ap_name] = float(kept_mean_ap)

      # For sanity checks, we also add the total number of samples.
      results["count"] = float(labels.shape[0])

      return results

  # pytype: enable=attribute-error

  def remove_skipped_classes(
      self,
      labels: Int["*b n"],
      skip_classes: Sequence[int],
  ) -> tuple[Int["*b n"], Bool["*b"]]:
    """Remove skipped classes from the labels and predictions."""
    keep_mask = np.ones(labels.shape[1], dtype=bool)
    keep_mask[list(skip_classes)] = False

    kept_labels = labels[:, keep_mask]

    return kept_labels, keep_mask

  @property
  def skip_classes_mean_ap_name(self) -> str:
    if self.class_names:
      skipped_class_names = [self.class_names[i] for i in self.skip_classes]
    else:
      skipped_class_names = [str(i) for i in self.skip_classes]
    skipped_classes_string = ",".join(skipped_class_names)
    return f"mean_AP_when_skipping_classes:{skipped_classes_string}"

  @typechecked
  def get_state(
      self,
      predictions: Float["*b n"],
      labels: Int["*b n"],
  ) -> MeanAveragePrecision.State:
    return self.State(
        labels=labels,
        predictions=predictions,
    )

  def __metric_names__(self) -> list[str]:
    # Add aggregated metric names.
    metric_names = ["mean_AP", "count"]

    # Add per-class metric name if class names are provided.
    if self.class_names:
      for class_name in self.class_names:
        metric_names.append(f"{class_name}_AP")
    if self.skip_classes:
      metric_names.append(self.skip_classes_mean_ap_name)
    return metric_names
