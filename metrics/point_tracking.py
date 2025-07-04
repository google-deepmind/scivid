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

"""Evaluation metrics for point tracking."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

from absl import logging
import flax
import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.metrics import base, base_state  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.typing import Float, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


def min_distance_to_ground_truth_for_each_pred(
    pred_endpoints: Float["Q 2"],
    pred_mask: Float["Q 2"],
    gt_endpoints: Float["Q 2"],
    gt_mask: Float["Q 2"],
) -> Float["Q"]:
  """Finds the distance of the closest point in gt_endpoints for each point in pred_endpoints.

  Args:
    pred_endpoints: Tracked endpoints (masked to size for batching)
    pred_mask: Array duplicated in [x,y] for an integer mask of pred_endpoints
    gt_endpoints: ground truth endpoints of point track (masked to size for
      batching)
    gt_mask: Array duplicated in [x,y] for an integer mask of gt_endpoints

  Returns:
    A jnp array of shape [Q] representing the min distance for each predicted
    keypoint to a ground truth endpoint.

  This is based off the original implementation from the STIR paper code:
  https://github.com/athaddius/STIRMetrics/blob/f425df12783ec8203840a861b81eecceaf3518b2/src/datatest/testutil.py#L55
  """
  bool_mask = gt_mask.astype(jnp.bool)

  # Big number should get masked out in the min calculation in distance_calc
  gt_endpoints = jnp.where(
      bool_mask,
      gt_endpoints,
      jnp.full_like(gt_endpoints, 1e15, dtype=jnp.float32),
  )

  def distance_calc(
      pred_endpoints: Float["Q 2"],
      pred_mask: Float["Q 2"],
      gt_endpoints: Float["Q 2"],
  ) -> Float["Q"]:
    # Compute the pairwise distances between the points in both arrays
    pairwise_distances: Float["Q Q"] = jnp.linalg.norm(
        gt_endpoints[:, None] - pred_endpoints[None], axis=-1
    )  # gt many rows, pred many cols

    # Find the distance of the closest point in gt for each point in pred
    min_distances: Float["Q"] = jnp.min(pairwise_distances, axis=0)

    # Mask out the distances for invisible predicted point tracks
    masked_min_distances: Float["Q"] = min_distances * pred_mask[:, 0]

    return masked_min_distances

  def no_ground_truth_inf_distances(
      pred_endpoints: Float["Q 2"],
      pred_mask: Float["Q 2"],  # pylint: disable=unused-argument
      gt_endpoints: Float["Q 2"],  # pylint: disable=unused-argument
  ) -> Float["Q"]:
    """Returns array of shape [Q] with inf values when no ground truth is provided.

    Args:
      pred_endpoints: Tracked endpoints.
      pred_mask: Masks with 1 values for valid predicted endpoints.
      gt_endpoints: Ground truth track endpoints.
    Returns a jnp array of infs to use where no ground truth is provided
    in the false branch of jax.lax.cond.
    This is needed for batching with vmap on statically sized inputs.
    """
    # Only the max number of queries is used in the false branch of
    # jax.lax.cond. All other inputs are unused.
    del gt_endpoints
    del pred_mask
    return jnp.full(shape=pred_endpoints.shape[0], fill_value=jnp.inf)

  return jax.lax.cond(
      jnp.any(bool_mask),
      distance_calc,  # Function if video has queries (distance for each pred)
      # Function if video has no ground truths (return inf distances)
      no_ground_truth_inf_distances,
      pred_endpoints,
      pred_mask,  # Mask distance calculated from invisible tracked points
      gt_endpoints,
  )


@dataclasses.dataclass
class StirTrackInfo:
  """Class storing the information for the tracks of a STIR video.

  Attributes:
    min_distances: A jnp array of shape [B Q] representing the min distance for
      each predicted point to a ground truth point.
    pred_endpoints: A jnp array of shape [B Q 2] representing the last frame
      positions of the predicted tracks.
    gt_endpoints: A jnp array of shape [B Q 2] representing the last frame
      positions of the ground truth tracks.
    pred_endpoint_masks: A jnp array of shape [B Q 1] representing the mask of
      the predicted tracks.
    gt_endpoint_masks: A jnp array of shape [B Q 1] representing the mask of the
      ground truth tracks.
    num_queries: The number of predicted tracks which matches the number of
      predicted points, as STIR eval assumes all query points remain visible.
  """

  min_distances: Float["*B Q"]
  pred_endpoints: Float["*B Q 2"]
  gt_endpoints: Float["*B Q 2"]
  pred_endpoint_masks: Float["*B Q 1"]
  gt_endpoint_masks: Float["*B Q 1"]

  @property
  def num_queries(self) -> Float[""]:
    """Returns the number of queried points.

    This matches the number of predicted points, as STIR eval assumes all query
    points remain visible.
    """
    # Get per-sample number of queries by summing across the query and mask
    # axis.
    return jnp.sum(self.pred_endpoint_masks, axis=(-2, -1))


def get_min_distances_to_gt_and_num_tracks_from_videos(
    pred_coords: Float["*B Q 2"],
    query_mask: Float["*B Q 1"],
    gt_coords: Float["*B Q 2"],
    target_mask: Float["*B Q 1"],
) -> StirTrackInfo:
  """Finds the distance of the closest point in gt_coords for each point in pred_coords."""

  # Duplicate mask for statically sized inputs for jax.lax.cond
  # masking for both [x,y] coordinates.
  pred_mask_xy: Float["*B Q 2"] = jnp.repeat(query_mask, 2, axis=-1)
  gt_mask_xy: Float["*B Q 2"] = jnp.repeat(target_mask, 2, axis=-1)

  # Find distance over each video in the batch
  min_distances = jax.vmap(
      min_distance_to_ground_truth_for_each_pred, in_axes=(0, 0, 0, 0)
  )(pred_coords, pred_mask_xy, gt_coords, gt_mask_xy)
  return StirTrackInfo(
      min_distances=min_distances,
      pred_endpoints=pred_coords,
      # Use query mask as predicted mask assumes all predicted points are
      # visible in last frame.
      pred_endpoint_masks=query_mask,
      gt_endpoints=gt_coords,
      gt_endpoint_masks=target_mask,
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class STIR2DErrorPerTrackAssumingNoOcclusions(base.Metric):
  """Based off of original implementation from STIR paper code.

  https://github.com/athaddius/STIRMetrics/blob/f425df12783ec8203840a861b81eecceaf3518b2/src/datatest/testutil.py#L55
  The competition reports 2D error per track, which is the average over
  predicted end points of the min distance to a ground truth point.

  NOTE: This assumes every tracked point that is queried is visible in the last
  frame.
  It also assumes the existence of at least one query per sample.

  Attributes:
    pred_coords: The predicted coordinates, a kontext.Key to a Float array of
      shape [B Q 2] representing the predicted tracks x and y coordinates.
    query_mask: The mask of the query points, a kontext.Key to a Float array of
      shape [B Q 1] with ones for the query points and zero padding.
    gt_coords: The ground truth coordinates, a kontext.Key to a Float array of
      shape [B Q 2] representing the ground truth tracks x and y coordinates.
    target_mask: The mask of the ground truth tracks, a kontext.Key to a Float
      array of shape [B Q 1] representing the mask of the ground truth tracks.
    rescale_height: The height to rescale the tracks to.
    rescale_width: The width to rescale the tracks to.
    round_to_nearest_pixel: Whether to round the tracks to the nearest pixel.
    thresholds: The pixel thresholds for accuracies.
    add_average_error: Whether to add the average error per track to the
      metrics.
  """

  pred_coords: kontext.Key = (
      kontext.REQUIRED
  )  # e.g. "pred.readouts.points.values"
  query_mask: kontext.Key = kontext.REQUIRED  # e.g. "batch.query_mask"
  gt_coords: kontext.Key = kontext.REQUIRED  # e.g. "batch.gt_tracks"
  target_mask: kontext.Key = (
      kontext.REQUIRED
  )  # e.g. "batch.target_points_visible"
  rescale_height: int = 1024
  rescale_width: int = 1280
  round_to_nearest_pixel: bool = True
  # pixel thresholds for accuracies
  thresholds: Sequence[int] = (4, 8, 16, 32, 64)
  add_average_error: bool = False

  @flax.struct.dataclass
  class State(base_state.CollectingState):
    """Collecting state which averages at the compute call.

    min_distances: A jnp array of shape [BN] representing the min distance for
      each predicted point to a ground truth point.
    num_queries: A jnp array of shape [BN] representing the number of
      queries, which equals the number of predicted points under the assumption
      that all queried points remain visible for each video.
    query_mask: A jnp array of shape [BN Q 1] representing the mask of the
      predicted tracks, with 1 values for valid queries.
    These are aggregated at the end to find the average error per *track*
    """

    min_distances: Float["num_videos Q"]
    num_queries: Float["num_videos"]
    query_masks: Float["num_videos Q 1"]

    @typechecked
    def compute(self) -> dict[str, float]:
      out = super().compute()
      min_distances = out.min_distances
      num_queries = out.num_queries
      query_masks = out.query_masks
      # Divide by num_queries to get average error per *track*
      if jnp.sum(num_queries) == 0:
        logging.warning(
            "No queries found in this batch, which will result in a infinite"
            " error in STIR2DErrorPerTrackAssumingNoOcclusions"
        )
      # The averaging is computed over the number of tracks across the entire
      # dataset (and not averaged per-sample).
      error: float = float(jnp.sum(min_distances) / jnp.sum(num_queries))

      result_dict = {}
      if self.parent.add_average_error:  # pytype: disable=attribute-error
        if min_distances.shape[0] == 0:
          error = 0.0
        result_dict["average_error"] = error

      # Compute per_threshold accuracies tracked in the STIR challenge
      # (see https://stir-challenge.github.io//evaluation/).
      per_threshold_accuracies = []
      for threshold in self.parent.thresholds:  # pytype: disable=attribute-error
        avg_threshold_accuracy = (
            # Count the number of points that are within the error threshold
            # among the valid predicted points.
            jnp.sum(min_distances <= threshold, where=query_masks[:, :, 0])
        ) / query_masks.sum()
        result_dict[f"within_{threshold}pxl_threshold"] = float(
            avg_threshold_accuracy
        )
        per_threshold_accuracies.append(avg_threshold_accuracy)
      result_dict["average_over_thresholds"] = float(
          np.mean(per_threshold_accuracies)
      )
      result_dict["num_queries"] = float(num_queries.sum())
      return result_dict

  def __metric_names__(self):
    # Add aggregated metric names.
    metric_names = [
        "average_over_thresholds",
        "num_queries",
    ]
    if self.add_average_error:
      metric_names.append("average_error")

    # Add per-threshold metric names.
    for threshold in self.thresholds:
      metric_names.append(f"within_{threshold}pxl_threshold")
    return metric_names

  @typechecked
  def get_state(
      self,
      pred_coords: Float["*B Q 2"],
      query_mask: Float["*B Q 1"],
      gt_coords: Float["*B Q 2"],
      target_mask: Float["*B Q 1"],
  ) -> STIR2DErrorPerTrackAssumingNoOcclusions.State:
    """Finds the distance of the closest point in gt_coords to each point in pred_coords.

    Args:
      pred_coords: A jnp array of shape [B Q 2] representing the predicted
        tracks x and y coordinates.
      query_mask: A jnp array of shape [B Q 1] with ones for the query points
        and zero padding.
      gt_coords: A jnp array of shape [B Q 2] representing the ground truth
        tracks.
      target_mask: A jnp array of shape [B Q 1] representing the mask of the
        ground truth tracks.

    Returns:
      A State object containing the min_distances and num_queries.
    """
    # Typing must be checked dynamically to allow for different shape
    # conventionsas as Union of Float annotations does not work to allow for
    # different axis shape verifications (the first annotation in the union
    # is used, instead of checking both annotations in the Union to see if one
    # is compatible).
    check_type(pred_coords, Float["*B Q 2"])
    check_type(gt_coords, Float["*B Q 2"])

    # For the STIR challenge, the errors are tracked in pixel space.
    scaling = jnp.array([self.rescale_width, self.rescale_height])
    gt_coords = gt_coords * scaling
    pred_coords = pred_coords * scaling
    if self.round_to_nearest_pixel:
      gt_coords = jnp.round(gt_coords)
      pred_coords = jnp.round(pred_coords)
    stir_track_info = get_min_distances_to_gt_and_num_tracks_from_videos(
        pred_coords,
        query_mask,
        gt_coords,
        target_mask,
    )

    return self.State(
        min_distances=stir_track_info.min_distances,
        num_queries=stir_track_info.num_queries,
        query_masks=stir_track_info.pred_endpoint_masks,
    )
