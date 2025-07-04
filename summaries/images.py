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

"""Image summaries."""

import dataclasses

import einops
from flax import struct
from jax import numpy as jnp
import kauldron as kd
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Array, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np

from scivid.summaries import utils as viz_utils


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowSubsampledImages(kd.summaries.ShowImages):
  """Show image summaries with optional reshaping and subsampling.

  Attributes:
    images: Key to the images to display.
    num_images: Number of images to collect and display. Default 5.
    vrange: Optional value range of the input images. Used to clip and then
      rescale the images to [0, 1].
    rearrange: Optional einops string to reshape the images.
    rearrange_kwargs: Optional keyword arguments for the einops reshape.
    subsample_dim: Optional dimension to subsample.
    subsample_step: Optional step size for subsampling.
  """

  subsample_dim: int | None = None
  subsample_step: int | None = None

  @typechecked
  def get_state(
      self,
      images: Float["..."],
  ) -> kd.summaries.ShowImages.State:
    images = _maybe_subsample(
        images, self.subsample_dim, step=self.subsample_step
    )
    return super().get_state(images)  # pytype: disable=attribute-error


def _maybe_subsample(
    array: Array["..."] | None,
    dimension: int | None = None,
    step: int | None = None,
) -> Array["..."] | None:
  """Subsamples the array along the given dimension with the given step.

  Args:
    array: The array to subsample.
    dimension: The dimension to subsample along.
    step: The subsampling step.

  Returns:
    The subsampled array.
  """
  if array is None or step is None:
    return array

  slices = [slice(None)] * array.ndim
  slices[dimension] = slice(None, None, step)
  return array[tuple(slices)]


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowTrackedPoints(metrics.Metric):
  """Show image summaries with tracked points.

  Attributes:
    videos: Key to the videos to display.
    tracks: Key to the tracks to display.
    visible: Key to the visibility of the tracks.
    display_frame: Index of the frame to display.
    num_videos: Number of samples to collect and display.
  """

  videos: kontext.Key = kontext.REQUIRED
  tracks: kontext.Key = kontext.REQUIRED
  visible: kontext.Key = kontext.REQUIRED

  display_frame: int = -1
  num_videos: int = 3

  def visualize_tracks(
      self,
      video: Float["T H W 3"],
      tracks: Float["Q T 2"],
      visible: Float["Q T 1"],
  ) -> Float["T H W 3"]:
    """Renders point tracks across a video."""
    frames = (255 * video).astype(np.uint8)
    occluded = 1 - visible.squeeze(-1)
    video_viz = viz_utils.plot_tracks(
        rgb=frames, points=tracks, occluded=occluded
    )
    video_viz = video_viz.astype(np.float32) / 255.0
    return video_viz

  @struct.dataclass
  class State(metrics.AutoState["ShowTrackedPoints"]):
    """Collects the data to display."""

    images: Float["n h w"] = metrics.truncate_field(
        num_field="parent.num_videos"
    )
    points: Float["n q 3"] = metrics.truncate_field(
        num_field="parent.num_videos"
    )
    visible: Float["n q 1"] = metrics.truncate_field(
        num_field="parent.num_videos"
    )

    @typechecked
    def compute(self) -> Float["n h w #3"]:
      results = super().compute()
      images = results.images
      points = results.points
      visible = results.visible
      # Display the points on each image.
      results = []
      for imgs, pts, vis in zip(images, points, visible):
        imgs_with_pts = self.parent.visualize_tracks(imgs, pts, vis)
        results.append(imgs_with_pts)
      results = jnp.stack(results, axis=0)[:, 0]  # skip the time dimension

      # always clip to avoid display problems in TB and Datatables
      return np.clip(results, 0.0, 1.0)

  @typechecked
  def get_state(
      self,
      videos: Float["B T H W 3"],
      tracks: Float["B T Q 2"],
      visible: Float["B T Q 1"],
  ) -> State:  # pytype: disable=signature-mismatch
    videos = videos[: self.num_videos]
    tracks = tracks[: self.num_videos]
    visible = visible[: self.num_videos]

    # Prepare the tracks for visualization.
    tracks = einops.rearrange(tracks, "... T Q C -> ... Q T C")
    visible = einops.rearrange(visible, "... T Q 1 -> ... Q T 1")

    # Bring back track coordinates to image space.
    time, height, width = videos.shape[-4:-1]
    tracks = tracks * np.array([width, height])

    # Handle negative frame indices by recovering the corresponding positive
    # index.
    if self.display_frame < 0:
      frame_idx = time + self.display_frame
    else:
      frame_idx = self.display_frame

    # Select frame to display, while keeping the time dimension.
    images = videos[:, frame_idx : frame_idx + 1]
    points = tracks[:, :, frame_idx : frame_idx + 1]
    points_visible = visible[:, :, frame_idx : frame_idx + 1]

    return self.State(images=images, points=points, visible=points_visible)
