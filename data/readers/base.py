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

"""Kauldron video reader."""

import abc
from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, ClassVar, Optional

from grain import python as grain
from kauldron.data.py import base
from kauldron.random import random
import tensorflow as tf

from scivid.data.readers import utils


Sampling = utils.Sampling


@dataclasses.dataclass(frozen=True, kw_only=True)
class PygrainVideoReader(base.DataSourceBase):
  """Simple pygrain-based abstract video reader class.

  Assumes that the data is stored as ArrayRecord files containing
  tf.SequenceExample protos.

  Attributes:
    dataset_name: Name of the dataset, must be set in the subclass.
    subset: Subset of the dataset to use - usually train, val, or test.
    split_paths: A dict mapping each subset name to where the corresponding
      ArrayRecord files are stored.
    ctx_feat: A dict describing the keys and formats of the context features,
      used by tf.io.parse_sequence_example to parse the context features.
    seq_feat: A dict describing the keys and formats of the sequence features,
      used by tf.io.parse_sequence_example to parse the sequence features.
    sampling: Sampling method to use, one of 'clip', 'random_clip',
      'multi_clip', 'middle_clip', or 'linspace_clip'.
      clip: Sample a single clip from the video starting at frame 0.
      random_clip: Sample a random clip from the video with random start frame.
      multi_clip: Sample multiple clips from the video, number of clips is
        defined by num_clips.
      middle_clip: Sample the middle clip with fixed stride eg. for single-clip
        fast eval.
      linspace_clip: Sample a clip from the video starting at frame 0 and ending
        at the last frame (using adaptive stride).
    num_frames: Number of frames to sample.
    stride: Stride for sampling frames.
    im_size: Resolution to resize the image to, if left as None, will not
      resize.
    num_clips: Number of clips to use, only used for multi clip sampling.
    input_video_key: Key of the image sequence in the sequence features.
    output_video_key: Key of the image in the output features.
  """

  dataset_name: str

  subset: str
  split_paths: ClassVar[dict[str, Any]] = None
  ctx_feat: ClassVar[dict[str, Any]] = {}
  seq_feat: ClassVar[dict[str, Any]] = {}

  sampling: str | Sampling = Sampling.CLIP
  num_frames: Optional[int] = None
  grayscale_to_rgb: bool = False
  stride: int = 1
  im_size: Optional[tuple[int, int]] = None
  resize_method: utils.ResizeMethod = (
      utils.ResizeMethod.INTERPOLATE_WITHOUT_ASPECT_RATIO
  )
  num_clips: int = 1
  input_video_key: str = 'image/encoded'
  output_video_key: str = 'image'

  def __post_init__(self):
    object.__setattr__(self, 'sampling', Sampling(self.sampling))

    if self.sampling == 'multi_clip' and self.num_clips <= 1:
      raise ValueError(
          'num_clips must be greater than 1 for multi clip sampling'
      )
    if self.sampling != 'multi_clip' and self.num_clips != 1:
      raise ValueError(
          'num_clips must be 1 for non multi clip sampling '
          f'but was set to: {self.num_clips}.'
      )

    if hasattr(super(), '__post_init__'):
      super().__post_init__()  # Future proof to run `__post_init__` in parents

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    path = self.split_paths[self.subset]
    return grain.ArrayRecordDataSource(path)

  def parse_sequence_example(
      self,
      serialized: tf.train.SequenceExample,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    context, sequence, _ = tf.io.parse_sequence_example(
        serialized, self.ctx_feat, self.seq_feat
    )
    return context, sequence

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    ds = super().ds_for_current_process(rng)

    # Load the TF sequence example.
    ds = ds.map(
        lambda serialized: self.parse_sequence_example(serialized=serialized),
    )

    # Perform dataset-specific preprocessing of each dataset element.
    ds = ds.map(
        lambda inputs: self.preprocess_element(*inputs),
    )

    return ds

  # Typically called by the subclass's preprocess_element method.
  def sample_clip(
      self,
      sequence: dict[str, Any],
  ) -> dict[str, Any]:
    """Samples a clip according to sampling arguments."""
    data_dict = {}
    vid_len = len(sequence[self.input_video_key])
    data_dict['vid_len'] = vid_len
    data_dict.update(
        **utils.process_idx(
            vid_len=vid_len,
            output_name='frame_idx',
            num_frames=self.num_frames,
            stride=self.stride,
            sampling=self.sampling,
            num_clips=self.num_clips,
        )
    )
    data_dict.update(
        **utils.process_video(
            sequence,
            input_name=self.input_video_key,
            output_name=self.output_video_key,
            frame_idx=data_dict['frame_idx'],
            num_frames=self.num_frames,
            im_size=self.im_size,
            resize_method=utils.ResizeMethod.INTERPOLATE_WITHOUT_ASPECT_RATIO,
            im_channels=3,
            dtype=tf.uint8,
            num_clips=self.num_clips,
            grayscale_to_rgb=self.grayscale_to_rgb,
        )
    )
    return data_dict

  @abc.abstractmethod
  def preprocess_element(
      self,
      context: dict[str, Any],
      sequence: dict[str, Any],
  ) -> dict[str, Any]:
    """Preprocesses a dataset element - should be implemented by subclass."""
    raise NotImplementedError()


def generate_array_paths(
    record_template: str, split: str, num_shards: int = 100
) -> Sequence[str]:
  """Generates a list of sharded file paths.

  This function is typically used when the data loading library requires an
  explicit list of all shard file paths, rather than a glob pattern.

  Args:
    record_template: A string template for the file paths. It should contain
      placeholders for `{split}`, `{shard_idx}`, and `{num_shards}`. For
      instance
      "path/to/data/{split}/prefix-{shard_idx:05d}-of-{num_shards:05d}.array_record"
    split: The dataset split name (e.g., "train", "test").
    num_shards: The total number of shards for this split.

  Returns:
    A list of formatted file path strings for all shards.
  """
  return [
      record_template.format(
          split=split, shard_idx=shard_idx, num_shards=num_shards
      )
      for shard_idx in range(num_shards)
  ]
