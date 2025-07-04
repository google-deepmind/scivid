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

"""Calms21 video reader for Kauldron.

Mouse behaviour multi-label classification dataset.

Dataset page: https://data.caltech.edu/records/zrznw-w7386.
"""

import dataclasses
import os
import pathlib
from typing import Any

import tensorflow as tf

from scivid.data.readers import base

# CalMS21 has 4 classes.
CLASSES = ('attacking', 'sniffing', 'mounting', 'not interacting')
NUM_CLASSES = len(CLASSES)

# Number of samples per class in the train set when dropping the last
# incomplete 32-sample batch.
# Note: the below counts are for our "slim" version of the train set, which we
# subsampled by 16x compared with the original train set. This is because in the
# original version of the dataset, clips are extracted with a stride of 1 in the
# time dimension, resulting in important redundancies in input frames.
LABEL_COUNTS_SLIM_TRAIN_DROP_LAST = (
    655,
    7586,
    1702,
    16681,
)
# number of samples in train when dropping the last incomplete 32-sample batch
NUM_SAMPLES_SLIM_TRAIN_DROP_LAST = 26_624
# The original full train set has the following counts:
# total: 430_592 / attacking: 10_690 / sniffing: 123_017 / mounting: 27_629 /
# not interacting: 269_256.


# Path to the SciVid data folder.
default_scivid_dir = pathlib.Path.home() / 'data/scivid'
scivid_data_dir = (
    pathlib.Path(os.getenv('SCIVID_DATA_DIR') or default_scivid_dir)
)
ARRAY_RECORD_TEMPLATE = str(
    scivid_data_dir /
    'full/calms21_slim/calms21_{split}_split_ids_shuffle_scale05.array_record-{shard_idx:05d}-of-{num_shards:05d}'
)
ARRAY_RECORD_TRAIN = str(
    scivid_data_dir /
    'full/calms21_slim/calms21_{split}_split_ids_shuffle_scale05_downsample16.array_record-{shard_idx:05d}-of-{num_shards:05d}'
)
# Both formats store tf.SequenceExample with the following features:
CTX_FEAT = {'clip/label/index': tf.io.VarLenFeature(dtype=tf.int64)}
SEQ_FEAT = {'image/encoded': tf.io.FixedLenSequenceFeature((), dtype=tf.string)}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Calms21PygrainReader(base.PygrainVideoReader):
  """Calms21 pygrain-based video reader for Kauldron."""

  dataset_name: str = 'calms21'
  split_paths = {
      # test set downscaled to 50% of the original size, same size as original
      # test set: 262_107 samples.
      'downscaled_test': base.generate_array_paths(
          ARRAY_RECORD_TEMPLATE, 'test', num_shards=1_024
      ),
      # 76_585 val samples downscaled to 50% of the original size.
      'downscaled_val': base.generate_array_paths(
          ARRAY_RECORD_TEMPLATE, 'val', num_shards=1_024
      ),
      # train set downscaled to 50% of the original size and downsampled by 16x
      # in the time dimension, resulting in 26_970 samples.
      'slim_train': base.generate_array_paths(
          ARRAY_RECORD_TRAIN, 'train', num_shards=1_024
      ),
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
    # Load label.
    data_dict.update({'label': tf.sparse.to_dense(context['clip/label/index'])})
    # Label frequencies over the slim train set are used to measure the
    # performance of the control (label frequency) baseline (note: they don't
    # exactly sum to one because certain samples have more than one label.)
    label_frequencies_train = (
        tf.constant(LABEL_COUNTS_SLIM_TRAIN_DROP_LAST)
        / NUM_SAMPLES_SLIM_TRAIN_DROP_LAST
    )
    data_dict.update({'label_frequencies_train': label_frequencies_train})
    return data_dict
