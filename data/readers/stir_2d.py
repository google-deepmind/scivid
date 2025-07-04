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

"""STIR 2D tracking video reader for Kauldron.

Data from Endoscopic videos. Note that videos are resized to half the resolution

Paper: https://arxiv.org/abs/2309.16782
Metrics Code: https://github.com/athaddius/STIRMetrics/tree/main,
Loader Code: https://github.com/athaddius/STIRLoader/tree/main
Data Set: https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared
"""

import dataclasses
import os
import pathlib
from typing import Any
from kauldron.typing import Float
import tensorflow as tf
from scivid.data.readers import base


# Max number of points across the dataset used for padding query and target
# point info to the same length.
_MAX_POINTS = 34

# Value to use for padding keypoint locations.
_PAD_VALUE = -1.0

# Path to the SciVid data folder.
default_scivid_dir = pathlib.Path.home() / 'data/scivid'
scivid_data_dir = (
    pathlib.Path(os.getenv('SCIVID_DATA_DIR') or default_scivid_dir)
)
ARRAY_RECORD_TEMPLATE = str(
    scivid_data_dir /
    'full/stir/{split}_stir.array_record-{shard_idx:05d}-of-{num_shards:05d}'
)

# Both formats store tf.SequenceExample with the following features:
CTX_FEAT = {
    'startir/object/keypoint/x': tf.io.VarLenFeature(dtype=tf.float32),
    'startir/object/keypoint/y': tf.io.VarLenFeature(dtype=tf.float32),
    'endir/object/keypoint/x': tf.io.VarLenFeature(dtype=tf.float32),
    'endir/object/keypoint/y': tf.io.VarLenFeature(dtype=tf.float32),
    'vid_idx': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
    'duration': tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
}
SEQ_FEAT = {'image/encoded': tf.io.FixedLenSequenceFeature((), dtype=tf.string)}


def _process_points(context: dict[str, Any]):
  """Processes points from context."""
  data_dict = {}
  query_coords = tf.stack(
      [
          tf.sparse.to_dense(context['startir/object/keypoint/x']),
          tf.sparse.to_dense(context['startir/object/keypoint/y']),
      ],
      axis=-1,
  )
  target_coords = tf.stack(
      [
          tf.sparse.to_dense(context['endir/object/keypoint/x']),
          tf.sparse.to_dense(context['endir/object/keypoint/y']),
      ],
      axis=-1,
  )
  # Pad keypoints to the max number of keypoints across data set.
  query_len = tf.shape(query_coords)[0]
  target_len = tf.shape(target_coords)[0]

  # Padding to allow batching
  query_coords: Float['Q 2'] = tf.pad(
      query_coords,
      [[0, _MAX_POINTS - query_len], [0, 0]],
      constant_values=_PAD_VALUE,
  )
  target_coords: Float['Q 2'] = tf.pad(
      target_coords,
      [[0, _MAX_POINTS - target_len], [0, 0]],
      constant_values=_PAD_VALUE,
  )

  # A mask indicating valid query points, padded to the max number of points.
  query_mask: Float['Q 1'] = tf.concat(
      [
          tf.ones(shape=(query_len, 1)),
          tf.zeros(shape=(_MAX_POINTS - query_len, 1)),
      ],
      axis=0,
  )

  # A mask indicating valid target points, padded to the max number of points.
  target_mask: Float['Q 1'] = tf.concat(
      [
          tf.ones(shape=(target_len, 1)),
          tf.zeros(shape=(_MAX_POINTS - target_len, 1)),
      ],
      axis=0,
  )
  query_mask = tf.ensure_shape(query_mask, (_MAX_POINTS, 1))
  target_mask = tf.ensure_shape(target_mask, (_MAX_POINTS, 1))
  query_coords = tf.ensure_shape(query_coords, (_MAX_POINTS, 2))
  target_coords = tf.ensure_shape(target_coords, (_MAX_POINTS, 2))
  data_dict.update({
      'query_coords': query_coords,
      'target_coords': target_coords,
      'query_mask': query_mask,
      'target_mask': target_mask,
  })
  return data_dict


@dataclasses.dataclass(frozen=True, kw_only=True)
class STIR2DPygrainReader(base.PygrainVideoReader):
  """STIR 2D pygrain-based video reader for Kauldron."""

  dataset_name: str = 'stir_2d'
  output_video_key: str = 'video'
  split_paths = {
      # Validation videos with durations <4 minutes.
      'valid': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'short_val'),
      # Full validation set (including long >10 minutes videos).
      'full': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'full_val'),
      # Test set.
      'test': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'test'),
  }
  ctx_feat = CTX_FEAT
  seq_feat = SEQ_FEAT

  def preprocess_element(
      self,
      context: dict[str, Any],
      sequence: dict[str, Any],
  ) -> dict[str, Any]:
    # Sample a clip.
    data_dict = self.sample_clip(sequence=sequence)
    # Add keypoint information.
    data_dict.update(_process_points(context))
    # Add video metadata.
    data_dict.update(
        {'video_idx': context['vid_idx'], 'duration': context['duration']}
    )
    return data_dict
