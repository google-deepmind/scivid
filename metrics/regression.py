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

"""Metrics for V4S2 regression tasks."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses

import flax.struct
from jax import numpy as jnp
from kauldron import kontext
from kauldron.losses import base as loss_base  # pylint: disable=g-importing-member
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Float  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member


# 0-indexed time steps used for computing the final score in Table 5 of the
# original paper https://arxiv.org/abs/2311.02665.
TYPHOON_EVAL_TIME_STEPS = (0, 1, 2, 5, 11)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TyphoonTemporalRegressionMetrics(base.Metric):
  r"""Reference metrics computed in Digital Typhoon forecasting task.

  Following the original implementation at
  https://github.com/kitamoto-lab/benchmarks/blob/1bdbefd7c570cb1bdbdf9e09f9b63f7c22bbdb27/forecasting/evaluate_forecasting_pipeline.py#L29-L33
  we compute per-timestep averaged absolute errors, relative absolute errors and
  root mean squared errors for per-timestep scalar predictions, typically
  pressure or wind speed.

  Attributes:
    preds: The predictions.
    targets: The targets.
    pred_len: The length of the predictions.

  Returns:
    A dictionary of metrics with the following convention:
    - for metrics computed at a given timestep:
    {variable_name}/t={time_step}/{metric_name}
    - for time-aggregated metrics:
    {variable_name}/{metric_name}
  """

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  pred_len: int = 12

  @flax.struct.dataclass
  class State(base_state.CollectingState):
    """RootMeanSquaredError state."""

    preds: Float["b t"]
    targets: Float["b t"]

    @typechecked
    def compute(self) -> dict[str, float]:
      out = super().compute()
      preds = out.preds
      targets = out.targets
      abs_errors: Float["b t"] = jnp.abs(preds - targets)
      rel_errors: Float["b t"] = abs_errors / targets
      squared_errors: Float["b t"] = jnp.square(preds - targets)
      # Compute RMSE.
      rmses = jnp.sqrt(jnp.mean(squared_errors, axis=0))

      # Average abolute and relative errors across samples.
      averaged_abs_errors: Float["t"] = jnp.mean(abs_errors, axis=0)
      averaged_relative_abs_errors: Float["t"] = jnp.mean(rel_errors, axis=0)

      # Return metrics for each timestep.
      result_metrics = {}

      # Digital Typhoon computes metrics for each of the 12 timesteps.
      for pred_step in range(self.parent.pred_len):  # pytype: disable=attribute-error
        result_metrics[f"pressure/t={pred_step}/abs_error"] = float(
            averaged_abs_errors[pred_step]
        )
        result_metrics[f"pressure/t={pred_step}/reative_abs_error"] = float(
            averaged_relative_abs_errors[pred_step]
        )
        result_metrics[f"pressure/t={pred_step}/rmse"] = float(rmses[pred_step])
      # Compute averaged RMSE across selected timesteps.
      time_steps = jnp.asarray(TYPHOON_EVAL_TIME_STEPS)
      result_metrics["pressure/rmse"] = float(jnp.mean(rmses[time_steps]))
      return result_metrics

  def __metric_names__(self) -> list[str]:
    return ["pressure/rmse"]

  @typechecked
  def get_state(
      self,
      preds: Float["b t"],
      targets: Float["b t"],
  ) -> TyphoonTemporalRegressionMetrics.State:

    return self.State(
        preds=preds,
        targets=targets,
    )


def compute_area_weights(
    grid_resolution: float = 1.0,
) -> Float["b t lat lon c"]:
  """Computes area weights for Weatherbench depending on the grid resolution.

  Args:
    grid_resolution: The grid resolution in degrees.

  Returns:
    Area weights with shape (1, 1, lat, 1, 1)
  """

  # We compute the weights following eq(1) https://arxiv.org/pdf/2308.15560.
  # following weather's _weight_for_latitude_vector_uniform_with_poles.
  # Note that we use this because our data readers set include_poles to True,
  # following GenCast and GraphCast configs.

  # Difference between upper & lower latitude cell bounds:
  d_lat = grid_resolution

  # Cell upper latitudes bounds are
  #   [90, 90-d_lat/2, 98-d_lat/2, ... -90+d_lat/2]
  # Cell lower latitudes bounds are
  #   [90-d_lat/2, 98-d_lat/2, ..., -90+d_lat/2, -90]
  # We consider extreme and regular cells separately.
  # Extreme values: trig. identity for (90, 90 - d_lat/2) (and symmetric).
  extreme_weights = (jnp.sin(jnp.deg2rad(d_lat / 4)) ** 2)[None]

  # Regular cells: sin(upper_lat) - sin(lower_lat); using trig. identity:
  latitudes = jnp.arange(-90, 90 + d_lat, d_lat)[1:-1]
  weights = jnp.cos(jnp.deg2rad(latitudes)) * jnp.sin(jnp.deg2rad(d_lat / 2))

  # Combine, normalize and add batch, time, longitude and channel dimensions.
  weights = jnp.concatenate([extreme_weights, weights, extreme_weights])
  weights = weights / weights.mean()
  weights = jnp.expand_dims(weights, axis=(0, 1, 3, 4))

  return weights


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Weatherbench2AreaWeightedRMSE(base.Metric):
  """Weatherbench 2 area-weighted RMSE (main metric)."""

  pred: kontext.Key = kontext.REQUIRED
  target: kontext.Key = kontext.REQUIRED
  grid_resolution: float = 1.0
  # Names of the selected variables and levels modeled (following
  # SELECTED_VARIABLES_AND_LEVELS in scenes4d/datasets/readers/weather.py)
  selected_variables_and_levels: tuple[str, ...] = (
      "Z500",
      "T850",
      "Q700",
  )
  score_label: str = "rmwse"

  @flax.struct.dataclass
  class State(base_state.AverageState):
    """Per-lead-time and per-channel average state."""

    @classmethod
    def from_values(
        cls,
        values: Float["b t c"],
    ) -> Weatherbench2AreaWeightedRMSE.State:
      """Factory to create the per-time and per-channel state from an array."""
      # count number of elements in the batch
      count = jnp.ones_like(values[..., 0, 0]).sum(dtype=jnp.float32)
      return cls(total=values.sum(0), count=count)

    def compute_weatherbench_aggregated_scores(
        self,
        metric_data: Mapping[str, float],
        selected_variables: tuple[str, ...],
        time_steps: int,
        *,
        metric_template: str = "{var_name}/t={t}/rmwse",
        score_label: str = "rmwse",
    ) -> Mapping[str, float]:
      """Compute the Weatherbench metrics.

      Compute scores, per-variable and aggregated across all
      variables.

      Args:
        metric_data: A dictionary {metric_name: metric_value}.
        selected_variables: Tuple of variable names.
        time_steps: Number of time steps.
        metric_template: The string template for the metric name, with
          placeholders for variable name and time step.
        score_label: Name of score label in metric data.

      Returns:
        A mapping containing the scores. Both per-variable and
        aggregated scores are included.
      """
      # Initialize per-variable & per-timestep lists for scores.
      scores = []

      weatherbench_scores = {}
      for var_name in selected_variables:
        var_scores = []
        for t in range(time_steps):
          metric_name = metric_template.format(var_name=var_name, t=t)
          model_perf = metric_data[metric_name]
          scores.append(model_perf)

          var_scores.append(model_perf)

        weatherbench_scores[f"{var_name}/{score_label}"] = float(
            jnp.mean(jnp.array(var_scores))
        )

      weatherbench_scores[score_label] = float(
          jnp.mean(jnp.array(scores))
      )
      return weatherbench_scores

    @typechecked
    def compute(self) -> dict[str, float]:
      mwse = super().compute()
      rmwse = jnp.sqrt(mwse)
      # Return metrics for each timestep.
      result_metrics = {}

      target_steps = rmwse.shape[0]

      # Store metrics separately for each timestep.
      selected_variables_and_levels = self.parent.selected_variables_and_levels  # pytype: disable=attribute-error
      for var_index, var_name in enumerate(selected_variables_and_levels):
        for target_step in range(target_steps):
          # Root mean weighted square errors following WB2 eq(1).
          result_metrics[f"{var_name}/t={target_step}/rmwse"] = float(
              rmwse[target_step, var_index]
          )

      aggregated_scores = self.compute_weatherbench_aggregated_scores(
          result_metrics,
          selected_variables=selected_variables_and_levels,
          time_steps=target_steps,
          score_label=self.parent.score_label,  # pytype: disable=attribute-error
      )
      score = aggregated_scores[
          self.parent.score_label  # pytype: disable=attribute-error
      ]
      if not jnp.allclose(score, float(rmwse.mean())):
        raise ValueError(
            f"Aggregated score {score} does not match"
            f" RMWSE averaged across time steps and variables {rmwse.mean()} "
        )

      result_metrics.update(aggregated_scores)
      return result_metrics

  @typechecked
  # Note: consider adding cached_property to this function to cache the weights.
  def area_weights(self) -> Float["b t lat lon c"]:
    return compute_area_weights(self.grid_resolution)

  @typechecked
  def get_state(
      self,
      pred: Float["b t lat lon c"],
      target: Float["b t lat lon c"],
  ) -> Weatherbench2AreaWeightedRMSE.State:

    square_errors = jnp.square(pred - target)

    # Compute weighted square errors (b x t x lat x lon x c).
    weights = self.area_weights().astype(square_errors.dtype)
    wse = weights * square_errors

    # Return per-lead-time and per-channel average weighted square errors.
    wse_per_lead_time_and_channel = jnp.mean(wse, axis=(-2, -3))  # (b x t x c)
    state = self.State.from_values(values=wse_per_lead_time_and_channel)

    return state

  def __metric_names__(self) -> list[str]:
    base_metrics = [self.score_label]
    for var_name in self.selected_variables_and_levels:
      base_metrics.append(f"{var_name}/{self.score_label}")
    return base_metrics


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Weatherbench2AreaWeightedL1(loss_base.Loss):
  """Weatherbench 2 area-weighted L1 loss.

  Can be used for training readouts on the weatherbench eval.

  Attributes:
    pred: Predicted values.
    target: Target values.
    residual_std: The standard deviation of residuals between consecutive time
      steps used to weigh the loss per channel.
    grid_resolution: The grid resolution in degrees.
  """

  pred: kontext.Key = kontext.REQUIRED
  target: kontext.Key = kontext.REQUIRED
  residual_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
  grid_resolution: float = 1.0

  @typechecked
  # Note: consider adding cached_property to this function to cache the weights.
  def area_weights(self) -> Float["b t lat lon c"]:
    return compute_area_weights(self.grid_resolution)

  @typechecked
  def get_values(
      self,
      pred: Float["b t lat lon c"],
      target: Float["b t lat lon c"],
  ) -> Float["*a"]:
    l1_errors = jnp.abs(pred - target)

    # Values are of different orders of magnitude across different channels.
    # Following GraphCast 4.2 loss, we weigh loss terms by the inverse standard
    # deviation of the time difference residuals.
    # This is equivalent to minimizing the errors between the gt and predicted
    # normalized residuals, which are more balanced across channels.
    consecutive_residual_std = jnp.full_like(
        pred, fill_value=jnp.array(self.residual_std)
    )
    # We assume that the residual variance between time steps t and t+k is k
    # times the 1-step residual variance (from t to t+1), implying independence
    # between time steps.
    time_step_std_scaling = jnp.sqrt(
        jnp.arange(1, pred.shape[-4] + 1)[None, :, None, None, None]
    )
    inverse_residual_std = 1.0 / (
        consecutive_residual_std * time_step_std_scaling
    )

    # We also weigh the loss by the area weights.
    area_weights = self.area_weights().astype(l1_errors.dtype)

    weights = area_weights * inverse_residual_std
    # Compute weighted L1 errors (b x t x lat x lon x c).
    weighted_l1_errors = weights * l1_errors

    return weighted_l1_errors
