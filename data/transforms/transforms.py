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

"""Transforms for data processing."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, List

import einops
from etils import enp
from etils import epy
import flax.core
import jax
from kauldron import kd
from kauldron.typing import TfArray, TfFloat, TfInt, XArray, typechecked  # pylint: disable=g-importing-member,g-multiple-import

FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict

with epy.lazy_imports():
  import tensorflow as tf  # pylint: disable=g-import-not-at-top
  from transformers.models.videomae import image_processing_videomae  # pylint: disable=g-import-not-at-top


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Normalize(kd.data.ElementWiseTransform):
  """Normalize an element."""

  in_vrange: tuple[float, float] = (0.0, 255.0)
  normalize_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
  normalize_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

  dtype: Any = tf.float32

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    xnp = enp.lazy.get_xnp(element)
    dtype = enp.lazy.as_np_dtype(self.dtype)
    element = xnp.asarray(element, dtype=dtype)
    _, in_max = self.in_vrange
    element = element / in_max
    element = (element - self.normalize_mean) / self.normalize_std

    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ExtractInitialFixedLengthClip(kd.data.ElementWiseTransform):
  """Extracts a fixed-length clip from the beginning of a list of images."""

  num_frames: int

  @typechecked
  def map_element(self, sequence: List[XArray]) -> XArray:

    if self.num_frames <= 0:
      raise ValueError("Number of frames must be positive.")

    if not sequence:
      raise ValueError("Sequence must not be empty.")

    first_clip = sequence[: self.num_frames]

    xnp = enp.lazy.get_xnp(first_clip[0], strict=False)

    # Pad if needed
    padding_len = max(0, self.num_frames - len(sequence))
    if padding_len:
      padding_frame = xnp.zeros_like(sequence[0])
      first_clip = first_clip + [padding_frame] * padding_len

    first_clip = xnp.stack(first_clip)

    return first_clip


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class JaxImageResize(kd.data.ElementWiseTransform):
  """Resizes an element using jax.image.resize."""

  shape: tuple[int, ...]
  method: str = (
      # can be "nearest", "linear", "bilinear", "trilinear", "triangle",
      # "cubic", "bicubic", "tricubic", "lanczos3", "lanczos5"
      "bilinear"
  )

  def map_element(self, element: XArray) -> XArray:
    return jax.image.resize(element, shape=self.shape, method=self.method)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class OneHot(kd.data.ElementWiseTransform):
  """One-hot encodes a list of labels.

  Can be used for both single- and multi-class one-hot encoding.
  The labels sequence dimension corresponds to the number of labels
  for a single batch element (and not to the batch dimension).
  Eg. labels = [0, 1] with num_classes=3 will be one-hot encoded as [1, 1, 0].

  Attributes:
    num_classes: Length of the one-hot vector (how many classes).
  """

  num_classes: int

  def __post_init__(self):
    super().__post_init__()
    if self.num_classes <= 0:
      raise ValueError("Number of classes must be positive.")

  @typechecked
  def map_element(self, labels: TfInt["..."], **kwargs) -> TfFloat["C"]:

    if labels.shape.rank == 0:
      labels = tf.reshape(labels, (1,))

    # Below handles both single- and multi-class one-hot encoding.
    x = tf.scatter_nd(
        labels[:, None], tf.ones(tf.shape(labels)[0]), (self.num_classes,)
    )

    # in case of duplicate labels, we do not accumulate the counter
    # (eg. [0, 0, 2] with num classes=3 -> [1, 0, 1] and not [2, 0, 1])
    x = tf.minimum(x, 1)

    return x


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class HfPreprocess(kd.data.ElementWiseTransform):
  """Applies a HuggingFace preprocessing function to the video."""

  processor: image_processing_videomae.VideoMAEImageProcessor

  def map_element(
      self, element: TfFloat["... T H W C"]
      ) -> TfFloat["... T C H W"]:
    preprocessed = self.processor(list(element), return_tensors="pt")
    preprocessed_np = preprocessed["pixel_values"].detach().numpy()
    return tf.squeeze(preprocessed_np)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Repeat(kd.data.ElementWiseTransform):
  """Einops repeat on a single element.

  Mostly a wrapper around einops.repeat, but also supports basic types like
  int, float, lists and tuples (which are converted to a numpy array first).

  Example:

  ```
  cfg.train_ds = kd.data.tf.Tfds(
      ...
      transforms=[
          ...,
          kd.data.Repeat(key="image", pattern="h w c -> t h w c",
                         axes_lengths={"t": 6}),
      ]
  )
  ```

  Attributes:
    pattern: `einops.repeat` pattern, e.g. "b h w c -> b c (h w)"
    axes_lengths: a dictionary for specifying additional axis e.g. number of
      repeats or axis that cannot be inferred from the pattern and the tensor
      alone.
  """

  pattern: str
  axes_lengths: dict[str, int] = dataclasses.field(default_factory=FrozenDict)

  @typechecked
  def map_element(self, element: Any) -> XArray:
    # Ensure element is an array (and not a python builtin)
    # This is useful e.g. for pygrain pipelines because often "label" will be
    # int and not an array, yet one might want to reshape it.
    xnp = enp.lazy.get_xnp(element, strict=False)
    element = xnp.asarray(element)

    return einops.repeat(element, self.pattern, **self.axes_lengths)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RepeatFrames(kd.data.ElementWiseTransform):
  """Repeats frames so that they are divisible by `divisible_by`.

  For example, if the input is 11 frames and `divisible_by` is 5, then 15 sample
  points will be selected using the image resize operation's sampling grid and
  then rounded to the nearest integer.
  """

  divisible_by: int

  @typechecked
  def map_element(
      self, element: TfArray["*b T H W C"]
  ) -> TfArray["*b T2 H W C"]:
    # Tensorflow image resize only supports height and width dimensions so for
    # the time dimension we use gather.
    t = tf.shape(element)[-4]
    t2 = (
        tf.cast(tf.math.ceil(t / self.divisible_by), tf.int32)
        * self.divisible_by
    )
    indices = tf.image.resize(
        tf.reshape(tf.range(t), [1, -1, 1]), [1, t2], method="nearest"
    )[0, :, 0]
    return tf.gather(element, indices, axis=-4)
