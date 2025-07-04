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

"""Preprocessing transforms for point tracks."""

import dataclasses

import grain.python as grain
import tensorflow as tf


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MaskOccludedTracks(grain.MapTransform):
  """Add mask for point or box tracks that are occluded in a query frame."""

  tracks_key: str = "target_coords"
  visible_key: str = "target_vis"
  mask_key: str = "tracks_mask"
  mask_value: float = -1.0
  query_timestep: int = 0

  def map(self, features):
    # tracks.shape = (num_timesteps, num_tracks, num_feature_dims)
    tracks = features[self.tracks_key]

    # visible.shape = (num_timesteps, num_tracks)
    visible = features[self.visible_key]

    # Compute track mask and integrate if mask already exists.
    mask = visible[self.query_timestep][:, tf.newaxis]
    if self.mask_key in features:
      mask = features[self.mask_key] * mask

    visible_query_frame = tf.broadcast_to(
        visible[self.query_timestep : self.query_timestep + 1], visible.shape
    )

    # Mask all positions for tracks not visible in query frame.
    tracks = tf.where(
        visible_query_frame[:, :, tf.newaxis] == 0.0,
        self.mask_value * tf.ones_like(tracks),
        tracks,
    )

    # Mask all visibility values for tracks not visible in query frame.
    visible = visible_query_frame * visible

    features[self.tracks_key] = tracks
    features[self.visible_key] = visible
    features[self.mask_key] = mask

    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CombineTrackMasksForLoss(grain.MapTransform):
  """Integrate per-track and dense masks into a single mask for losses."""

  target_vis_key: str = "target_vis"
  query_mask_key: str = "query_mask"
  target_mask_key: str = "target_mask"

  def map(self, features):
    # target_vis.shape = (T, Q, 1)
    target_vis = features[self.target_vis_key]

    # query_mask.shape = (Q, 1)
    query_mask = features[self.query_mask_key]

    # target_mask.shape = (T, Q, 1)
    features[self.target_mask_key] = target_vis * query_mask[tf.newaxis, :, :]

    return features
