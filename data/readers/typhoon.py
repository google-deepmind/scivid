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

"""Digital Typhoon forecasting dataset.

Paper: https://arxiv.org/abs/2311.02665
"""

import dataclasses
import os
import pathlib

from typing import Any

import numpy as np
import tensorflow as tf

from scivid.data.readers import base
from scivid.data.readers import utils

# Forecasting takes 12 input frames and predicts 12 future frames.
NUM_INPUT_FRAMES = 12
PRED_LEN = 12

IMAGE_RESOLUTION = 256

# Pressure values, per time step, averaged across samples, for 24 first
# timesteps of training set.
AVERAGE_TRAIN_PRESSURES = [
    1004.0512931209871,
    1003.8195427204,
    1003.5840493563948,
    1003.3948277440564,
    1003.2083354379939,
    1002.9607740380299,
    1002.7633620514267,
    1002.5030193109621,
    1002.2218374493478,
    1001.9390804246924,
    1001.6788807134519,
    1001.357038826778,
    1001.1169542334545,
    1000.8285938131398,
    1000.5369237263998,
    1000.2145115644082,
    999.9189669729649,
    999.5570388443168,
    999.1941090726305,
    998.8002891102057,
    998.4185328867244,
    998.0522987979582,
    997.7094841441889,
    997.3081885151479,
]

# Pressure averaged across all time steps for all train sequences:
AVERAGE_TRAIN_PRESSURE = 983.9111856685796

# Path to the SciVid data folder.
default_scivid_dir = pathlib.Path.home() / 'data/scivid'
scivid_data_dir = (
    pathlib.Path(os.getenv('SCIVID_DATA_DIR') or default_scivid_dir)
)
ARRAY_RECORD_TEMPLATE = str(
    scivid_data_dir /
    'full/digital_typhoon_slim/{split}_imsize_256_max_frames_32.array_record-{shard_idx:05d}-of-{num_shards:05d}'
)

# Both formats store tf.SequenceExample with the following features:
CTX_FEAT = {'key_': tf.io.FixedLenFeature((), dtype=tf.string)}
SEQ_FEAT = {
    'image/infrared': tf.io.VarLenFeature(tf.float32),
    'pressure': tf.io.VarLenFeature(tf.float32),
    'wind': tf.io.VarLenFeature(tf.float32),
    'lat': tf.io.VarLenFeature(tf.float32),
    'lon': tf.io.VarLenFeature(tf.float32),
    'grade': tf.io.VarLenFeature(tf.int64),
    'year': tf.io.VarLenFeature(tf.int64),
    'month': tf.io.VarLenFeature(tf.int64),
    'day': tf.io.VarLenFeature(tf.int64),
    'hour': tf.io.VarLenFeature(tf.int64),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class TyphoonPygrainReader(base.PygrainVideoReader):
  """Typhoon pygrain-based video reader for Kauldron."""

  load_pressure: bool = False
  load_wind: bool = False
  load_lat: bool = False
  load_lon: bool = False
  load_grade: bool = False
  load_year: bool = False
  load_month: bool = False
  load_day: bool = False
  load_hour: bool = False

  num_frames: int = NUM_INPUT_FRAMES
  pred_len: int = PRED_LEN
  max_start_frame: int | None = None
  dataset_name: str = 'typhoon'
  split_paths = {
      'test': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'test', 50),
      'train': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'train', 50),
      'train_val': base.generate_array_paths(
          ARRAY_RECORD_TEMPLATE, 'train_val', 50
      ),
      'val': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'val', 50),
  }
  ctx_feat = CTX_FEAT
  seq_feat = SEQ_FEAT
  im_seq_key: str = 'image/infrared'

  @property
  def num_past_and_future_timesteps(self) -> int:
    return self.num_frames + self.pred_len

  def preprocess_element(
      self,
      context: dict[str, Any],
      sequence: dict[str, Any],
  ) -> dict[str, Any]:
    # Ensure eval is performed on the first frames.
    if self.subset == 'val' or self.subset == 'test':
      if self.sampling != 'clip':
        raise ValueError(
            'sampling must be clip (on first frames) for val and test, but got'
            f' sampling={self.sampling} for subset={self.subset} as in '
            'https://github.com/kitamoto-lab/benchmarks/blob/1bdbefd7c570cb1bdbdf9e09f9b63f7c22bbdb27/forecasting/Dataloader/PadSequence.py#L19-L20'
        )

    data_dict = {}
    image_name = 'image/infrared'
    arr = tf.sparse.to_dense(sequence[image_name])
    vid_len = arr.shape[0]
    data_dict['vid_len'] = np.array((vid_len), dtype=np.int32)

    # Select indices of frames for input and outputs. Note: we use
    # num_past_and_future_timesteps as num_frames here to select the indices
    # because we need to sample num_frames past frames and
    # pred_len future values consistently.
    data_dict.update(
        **utils.process_idx(
            vid_len=vid_len,
            output_name='frame_idx',
            num_frames=self.num_past_and_future_timesteps,
            stride=self.stride,
            sampling=self.sampling,
            num_clips=self.num_clips,
            max_start_frame=self.max_start_frame,
        )
    )

    # Adding input images.
    arr = tf.gather(tf.squeeze(arr), data_dict['frame_idx'])
    arr = tf.reshape(
        arr,
        (
            self.num_past_and_future_timesteps,
            IMAGE_RESOLUTION,
            IMAGE_RESOLUTION,
            1,
        ),
    )
    arr = arr[: self.num_frames]

    # Repeat channel dimension to 3 to match "RGB" format for inputs.
    arr = tf.tile(arr, [1, 1, 1, 3])

    data_dict.update({'video': arr})

    # Adding target variables for each batch (pressure, wind, etc.)
    for var in sequence.keys():
      if var != self.im_seq_key:
        load_var = getattr(self, f'load_{var}')
        if load_var:
          arr = tf.sparse.to_dense(sequence[var])
          arr = tf.gather(tf.squeeze(arr), data_dict['frame_idx'])
          arr = tf.reshape(arr, (self.num_past_and_future_timesteps,))
          # Only add the future values.
          data_dict.update(
              {var: arr[self.num_frames : self.num_past_and_future_timesteps]}
          )

        # Add last input pressure and wind values for control evaluations.
        if var in ['pressure', 'wind']:
          data_dict[f'last_{var}'] = arr[self.num_frames - 1]

    data_dict['video_id'] = tf.strings.to_number(
        context['key_'], out_type=tf.int64
    )

    # Average per-time-step future pressure over all training sequences used for
    # control baseline prediction, where the average pressure at each time step
    # is used as the baseline predictions.
    # (see 'predict_mean_pressures' in: typhoon_future_pred.py)
    data_dict['mean_future_pressures'] = tf.constant(
        AVERAGE_TRAIN_PRESSURES[
            self.num_frames : self.num_past_and_future_timesteps
        ],
        dtype=tf.float32,
    )

    # We calculate the average future pressure over all training sequences
    # and use it to normalize the output pressure values by subtracting
    # the average train pressure.
    # (see loss_targets in train_losses in: typhoon_future_pred.py)
    data_dict['global_mean_pressure'] = tf.constant(
        [AVERAGE_TRAIN_PRESSURE for _ in range(self.pred_len)],
        dtype=tf.float32,
    )
    data_dict['normalized_pressure'] = (
        data_dict['pressure'] - data_dict['global_mean_pressure']
    )
    return data_dict
