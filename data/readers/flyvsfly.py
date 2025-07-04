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

"""FlyVsFly fly behaviour classification dataset.

Dataset page: https://data.caltech.edu/records/zrznw-w7386
"""

import dataclasses

import os
import pathlib
from typing import Any

import tensorflow as tf

from scivid.data.readers import base

# Following VideoPrism (https://arxiv.org/pdf/2402.13217.pdf) the 10 original
# dataset labels are grouped into 7 classes:
CLASSES = (
    'lunging',
    'wing_threat',
    'tussling',
    'wing_extension',
    'circling',
    'copulation',
    # This background clas is not used in the final metric.
    'not_interacting',
)
# Number of samples per class in the train set when dropping the last
# incomplete 32-sample batch
LABEL_COUNTS_TRAIN_DROP_LAST = (
    6827,
    23497,
    1950,
    18239,
    4388,
    329372,
    687413,
)
# Number of samples in train when dropping the last incomplete 32-sample batch
NUM_SAMPLES_TRAIN_DROP_LAST = 1_067_008
NUM_CLASSES = len(CLASSES)


# Path to the SciVid data folder.
default_scivid_dir = pathlib.Path.home() / 'data/scivid'
scivid_data_dir = (
    pathlib.Path(os.getenv('SCIVID_DATA_DIR') or default_scivid_dir)
)
ARRAY_RECORD_TEMPLATE = str(
    scivid_data_dir /
    'full/fly_vs_fly/{split}_fly_split_ids_shuffle.array_record-{shard_idx:05d}-of-{num_shards:05d}'
)

# Both formats store tf.SequenceExample with the following features:
CTX_FEAT = {'clip/label/index': tf.io.VarLenFeature(dtype=tf.int64)}
SEQ_FEAT = {'image/encoded': tf.io.FixedLenSequenceFeature((), dtype=tf.string)}


@dataclasses.dataclass(frozen=True, kw_only=True)
class FlyvsFlyPygrainReader(base.PygrainVideoReader):
  """FlyvsFly pygrain-based video reader for Kauldron."""

  dataset_name: str = 'flyvsfly'
  split_paths = {
      # Same train and test data as was used in VideoPrism, but with shuffled
      # sstable keys.
      'test': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'test'),
      'train': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'train'),
      # Validation set is composed from 4 videos which were not used in
      # VideoPrism neither at train or test time.
      'val': base.generate_array_paths(ARRAY_RECORD_TEMPLATE, 'val'),
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
    # Label frequencies over the train set are used to measure the performance
    # of the control (label frequency) baseline (note: they don't exactly sum to
    # one because certain samples have more than one label.)
    label_frequencies_train = (
        tf.constant(LABEL_COUNTS_TRAIN_DROP_LAST) / NUM_SAMPLES_TRAIN_DROP_LAST
    )
    data_dict.update({'label_frequencies_train': label_frequencies_train})
    return data_dict
