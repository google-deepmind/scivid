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

"""MOVi video reader for Kauldron."""

import dataclasses
import os
import pathlib
from typing import Any

import tensorflow as tf

from scivid.data.readers import base
from scivid.data.readers import utils


# Path to the SciVid data folder.
default_scivid_dir = pathlib.Path.home() / 'data/scivid'
scivid_data_dir = (
    pathlib.Path(os.getenv('SCIVID_DATA_DIR') or default_scivid_dir)
)
ARRAY_RECORD_TEMPLATE = str(
    scivid_data_dir /
    'full/movi/{split}.array_record-{shard_idx:05d}-of-{num_shards:05d}'
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MOViPygrainReader(base.PygrainVideoReader):
  """MOVi pygrain-based video reader for Kauldron.
  """

  dataset_name: str = 'movi'
  input_video_key: str = 'image/encoded'
  output_video_key: str = 'video'
  num_tracks: int = 64

  split_paths = {
      'train': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'train'),
  }

  def __post_init__(self):
    super().__post_init__()

    ctx_feat = {}
    seq_feat = {
        'image/encoded': tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        'tracks/2d': tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        'tracks/visibility': tf.io.FixedLenSequenceFeature((), dtype=tf.string),
    }

    object.__setattr__(self, 'ctx_feat', ctx_feat)
    object.__setattr__(self, 'seq_feat', seq_feat)

  def preprocess_element(
      self,
      context: dict[str, Any],
      sequence: dict[str, Any],
  ) -> dict[str, Any]:
    # Sample a clip.
    data_dict = self.sample_clip(sequence=sequence)

    # Add labels.
    data_dict.update(
        **utils.process_uncompressed_tensor(
            sequence,
            input_name='tracks/2d',
            output_name='target_coords',
            frame_idx=data_dict['frame_idx'],
            tensor_shape=(None, 2),
            dtype=tf.float32,
            num_frames=self.num_frames,
            num_clips=self.num_clips,
        )
    )
    data_dict.update(
        **utils.process_uncompressed_tensor(
            sequence,
            input_name='tracks/visibility',
            output_name='target_vis',
            frame_idx=data_dict['frame_idx'],
            tensor_shape=(None,),
            dtype=tf.float32,
            num_frames=self.num_frames,
            num_clips=self.num_clips,
        )
    )

    # Sample random tracks.
    max_num_tracks = data_dict['target_coords'].shape[1]
    track_indices = tf.random.shuffle(tf.range(max_num_tracks))
    track_indices = track_indices[: self.num_tracks]
    data_dict['target_coords'] = tf.gather(
        data_dict['target_coords'], track_indices, axis=1
    )
    data_dict['target_vis'] = tf.gather(
        data_dict['target_vis'], track_indices, axis=1
    )

    return data_dict
