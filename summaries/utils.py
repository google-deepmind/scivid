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

"""Visualization utility functions."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_tracks(
    rgb: np.ndarray,
    points: np.ndarray,
    occluded: np.ndarray,
    gt_points: Optional[np.ndarray] = None,
    gt_occluded: Optional[np.ndarray] = None,
    trackgroup: Optional[np.ndarray] = None,
    point_size: int = 20,
) -> np.ndarray:
  """Plot tracks with matplotlib.

  This function also supports plotting ground truth tracks alongside
  predictions, and allows you to specify tracks that should be plotted
  with the same color (trackgroup).  Note that points which are out of
  bounds will be clipped to be within bounds; mark them as occluded if
  you don't want them to appear.

  Args:
    rgb: frames of shape [num_frames, height, width, 3].  Each frame is passed
      directly to plt.imshow.
    points: tracks, of shape [num_points, num_frames, 2], np.float32. [0, width
      / height]
    occluded: [num_points, num_frames], bool, True if the point is occluded.
    gt_points: Optional, ground truth tracks to be plotted with diamonds, same
      shape/dtype as points
    gt_occluded: Optional, ground truth occlusion values to be plotted with
      diamonds, same shape/dtype as occluded.
    trackgroup: Optional, shape [num_points], int: grouping labels for the
      plotted points.  Points with the same integer label will be plotted with
      the same color.  Useful for clustering applications.
    point_size: int, the size of the plotted points, passed as the 's' parameter
      to matplotlib.

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
  disp = []
  cmap = plt.cm.hsv  # pytype: disable=module-attr

  z_list = (
      np.arange(points.shape[0]) if trackgroup is None else np.array(trackgroup)
  )

  # random permutation of the colors so nearby points in the list can get
  # different colors
  z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
  colors = cmap(z_list / (np.max(z_list) + 1))
  figure_dpi = 64

  for i in range(rgb.shape[0]):
    fig = plt.figure(
        figsize=(rgb.shape[2] / figure_dpi, rgb.shape[1] / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w',
    )
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(rgb[i] / 255.0)
    colalpha = np.concatenate(
        [colors[:, :-1], 1 - occluded[:, i : i + 1]], axis=1
    )
    points = np.maximum(points, 0.0)
    points = np.minimum(points, [rgb.shape[2], rgb.shape[1]])
    plt.scatter(points[:, i, 0], points[:, i, 1], s=point_size, c=colalpha)
    occ2 = occluded[:, i : i + 1]
    if gt_occluded is not None:
      occ2 *= 1 - gt_occluded[:, i : i + 1]

    if gt_points is not None:
      gt_points = np.maximum(gt_points, 0.0)
      gt_points = np.minimum(gt_points, [rgb.shape[2], rgb.shape[1]])
      colalpha = np.concatenate(
          [colors[:, :-1], 1 - gt_occluded[:, i : i + 1]], axis=1
      )
      plt.scatter(
          gt_points[:, i, 0],
          gt_points[:, i, 1],
          s=point_size + 6,
          c=colalpha,
          marker='D',
      )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img_buf = fig.canvas.buffer_rgba()  # pytype: disable=attribute-error
    img = np.frombuffer(img_buf, dtype='uint8').reshape(
        int(height), int(width), 4
    )
    img = img[..., :3]  # Convert RGBA to RGB
    disp.append(np.copy(img))
    plt.close(fig)
    del fig, ax

  disp = np.stack(disp, axis=0)
  return disp
