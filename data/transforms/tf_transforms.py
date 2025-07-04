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

"""pygrain transforms processing tensorflow arrays used for SciVid paper experiments."""

import dataclasses
from typing import Literal, Optional
import einops
import grain.python as grain
from kauldron.data.transforms import base
from kauldron.typing import TfArray, typechecked  # pylint: disable=g-importing-member,g-multiple-import
import numpy as np
import tensorflow as tf


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseRandomTransform(
    base.ElementWiseRandomTransformBase, grain.RandomMapTransform
):
  """Base class for random elementwise transforms."""

  def random_map(self, features, rng: np.random.Generator):
    # Sample the seed *once* here before iterating through the elements.
    # This ensures that the same random parameters (e.g., crop offsets,
    # flip decision) are used for all specified keys ('self.key') within
    # a single example, maintaining consistency across related features
    # like 'video' and 'segmentations' or 'target_points'...
    seed = _get_tf_seeds(rng)
    features_out = {}
    for key, element, should_transform in self._per_element(features):
      if should_transform:
        features_out[key] = self.random_map_element(element, seed)
      else:
        features_out[key] = element
    return features_out


def _get_tf_seeds(rng: np.random.Generator) -> tf.Tensor:
  """Generate integer seeds from the NumPy generator for TensorFlow's stateless operations."""
  tf_seed1 = rng.integers(2**31 - 1, size=(), dtype="int64")
  tf_seed2 = rng.integers(2**31 - 1, size=(), dtype="int64")
  return tf.stack([tf_seed1, tf_seed2])


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomResize(ElementWiseRandomTransform):
  """Scales video randomly between a set of scale factors."""

  prob: float = 0.8
  min_scale_factor: float = 0.8
  max_scale_factor: float = 1.2
  method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR
  antialias: bool = False

  @typechecked
  def random_map_element(
      self, element: TfArray["*B H W C"], seed: tf.Tensor
  ) -> TfArray["*B h w C"]:
    seed1, seed2 = _split_seed(seed)
    scale = tf.random.stateless_uniform(  # same scale for height and width
        shape=[],
        seed=seed1,
        minval=self.min_scale_factor,
        maxval=self.max_scale_factor,
    )
    h = tf.cast(tf.shape(element)[-3], tf.float32)
    w = tf.cast(tf.shape(element)[-2], tf.float32)
    resize_height = tf.cast(scale * h, tf.int32)
    resize_width = tf.cast(scale * w, tf.int32)
    resized_element = tf.image.resize(
        element,
        (resize_height, resize_width),
        method=self.method,
        antialias=self.antialias,
    )
    coin_toss = tf.random.stateless_uniform(
        (), minval=0, maxval=1, dtype=tf.float32, seed=seed2
    )
    element = tf.cond(
        pred=tf.less(coin_toss, tf.cast(self.prob, tf.float32)),
        true_fn=lambda: resized_element,
        false_fn=lambda: element,
    )
    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomCrop(ElementWiseRandomTransform):
  """Randomly crop the input data to the specified shape.

  Can be used on data of any shape or type including images and videos.

  Attributes:
    shape: A tuple of integers describing the target shape of the crop. Entries
      can be also be None to keep the original shape of the data in that dim.
  """

  shape: tuple[Optional[int], ...]

  def random_map(self, features, rng: np.random.Generator):
    if not all([d is None or d >= 0 for d in self.shape]):
      raise ValueError(
          "Target shape can contain only non-negative ints or None. Got"
          f" {self.shape=}"
      )
    shapes = {k: v.shape for k, v in features.items() if k in self.key}
    for key, shape in shapes.items():
      if len(shape) != len(self.shape):
        raise ValueError(
            "Rank of self.shape has to match element shape. But got"
            f" {self.shape=} and {shape=} for {key!r}"
        )
    ref_key, ref_shape = next(iter(shapes.items())) if shapes else (None, None)
    # ensure dimensions match except where self.shape is None
    for key, shape in shapes.items():
      for ref_dim, key_dim, target_dim in zip(ref_shape, shape, self.shape):
        if ref_dim != key_dim and (target_dim is not None):
          raise ValueError(
              "Shapes of different keys for random crop have to be compatible,"
              f" but got {ref_shape} ({ref_key}) != {shape} ({key}) with"
              f" {self.shape=}"
          )

    return super().random_map(features, rng)

  @typechecked
  def random_map_element(
      self, element: TfArray["..."], seed: tf.Tensor
  ) -> TfArray["..."]:
    shape = tf.shape(element)
    # resolve dynamic portions of self.shape to a static target_shape
    target_shape = get_target_shape(element, self.shape)
    # compute the range of the offset for the tf.slice
    offset_range = shape - target_shape
    clipped_offset_range = tf.clip_by_value(offset_range, 1, tf.int32.max)

    # randomly sample offsets from the desired range via modulo
    rand_int = tf.random.stateless_uniform(
        [shape.shape[0]],
        seed=seed,
        minval=None,
        maxval=None,
        dtype=tf.int32,
    )
    offset = tf.where(offset_range > 0, rand_int % clipped_offset_range, 0)
    return tf.slice(element, offset, target_shape)  # crop


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TemporalRandomWindow(ElementWiseRandomTransform):
  """Gets a random slice (window) along 0-th axis of input tensor.

  Pads the input tensor along the time axis if the tensor length is shorter than
  the provided length. Supports padding with the last value or a constant value.

  Attr:
    length: An integer representing the desired length of the output tensor.
    padding_mode: Either "last" or "constant", specifying the padding strategy.
      "last" repeats the last value of the input tensor. "constant" pads with a
      constant value.
    padding_value: A float defining the value with which to pad when
      `padding_mode` is "constant".  Ignored if `padding_mode` is "last".
    frame_rate: The frame rate to use if `random_frame_rate` is False.
  """

  length: int
  padding_mode: Literal["constant", "last"] = "constant"
  padding_value: float = 0.0
  frame_rate: int | Literal["random"] = 1

  @typechecked
  def random_map_element(  # pylint: disable=arguments-renamed
      self, tensor: TfArray["T *C"], seed
  ) -> TfArray["t *C"]:
    length = tf.minimum(self.length, tf.shape(tensor)[0])

    rank = len(tensor.shape)
    seed1, seed2 = _split_seed(seed)
    if self.frame_rate == "random":
      max_frame_rate = tf.cast(tf.floor(tf.shape(tensor)[0] / length), tf.int32)
      frame_rate = tf.random.stateless_uniform(
          shape=[],
          seed=seed1,
          minval=1,
          maxval=max_frame_rate + 1,
          dtype=tf.int32,
      )
    else:
      frame_rate = self.frame_rate
    length = frame_rate * length
    window_size = tf.concat(([length], tf.shape(tensor)[1:]), axis=0)
    tensor = tf.image.stateless_random_crop(
        tensor, size=window_size, seed=seed2
    )
    indices = tf.range(start=0, limit=tf.shape(tensor)[0], delta=frame_rate)
    tensor = tf.gather(tensor, indices)
    frames_to_pad = tf.maximum(self.length - tf.shape(tensor)[0], 0)

    if self.padding_mode == "constant":
      tensor = tf.pad(
          tensor,
          ((0, frames_to_pad),) + ((0, 0),) * (rank - 1),
          constant_values=self.padding_value,
      )
    elif self.padding_mode == "last":
      padding = tf.tile(tensor[-1:], [frames_to_pad] + [1] * (rank - 1))
      tensor = tf.concat([tensor, padding], axis=0)
    else:
      raise ValueError(f"Unknown padding mode: {self.padding_mode}")
    tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
    return tf.cast(tensor, tensor.dtype)


def get_target_shape(t: tf.Tensor, target_shape):
  """Resolve the `dynamic` portions of `target_shape`."""
  finale_shape = []
  dynamic_shape = tf.shape(t)
  for i, (static_dim, target_dim) in enumerate(zip(t.shape, target_shape)):
    if target_dim is not None:
      finale_shape.append(target_dim)
    elif static_dim is not None:
      finale_shape.append(static_dim)
    else:
      finale_shape.append(dynamic_shape[i])
  return finale_shape


def _split_seed(rng: tf.Tensor, num_splits: int = 2) -> list[tf.Tensor]:
  """Splits the given random seed into several."""
  return tf.unstack(tf.random.split(rng, num=num_splits))


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GaussianBlur(ElementWiseRandomTransform):
  """Gaussian Blur.

  Apply gaussian blurring of standard deviation uniformly sampled between
  sigma_min and sigma_max with probability apply_prob.
  """

  apply_prob: float = 0.5
  sigma_min: float = 0.1
  sigma_max: float = 2.0
  kernel_size: int = 9

  @typechecked
  def random_map_element(
      self, element: TfArray["*B H W C"], seed: tf.Tensor
  ) -> TfArray["*B h w C"]:

    # Randomly sample a sigma value.
    # Sigma corresponds to the standard deviation of the Gaussian kernel.
    seed1, seed2 = _split_seed(seed)
    sigma = tf.random.stateless_uniform(
        [],
        seed1,
        minval=self.sigma_min,
        maxval=self.sigma_max,
        dtype=tf.float32,
    )

    # Converts kernel size into odd integer to ensure center pixel.
    kernel_size = 2 * int(self.kernel_size / 2) + 1

    # Creates a 1D kernel of that size and sets it to be a Gaussian.
    x = tf.cast(tf.range(-(kernel_size // 2), kernel_size // 2 + 1), tf.float32)
    blur_filter = tf.exp(-(x**2) / (2.0 * sigma**2))
    # Normalizes the kernel to sum to 1.
    blur_filter = blur_filter / tf.reduce_sum(blur_filter)

    # Creates 1D filters horizontally and vertically to achieve 2D
    # convolution. This works because the Gaussian is separable.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_v = tf.tile(blur_v, [1, 1, element.shape[-1], 1])
    blur_h = tf.transpose(blur_v, [1, 0, 2, 3])

    # Does the actual blurring using depthwise_conv2d.
    blurred = tf.nn.depthwise_conv2d(
        element, blur_h, strides=[1, 1, 1, 1], padding="SAME"
    )
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding="SAME"
    )

    # Randomly apply the blur based on apply_prob.
    coin_toss = tf.random.stateless_uniform(
        (), minval=0, maxval=1, dtype=tf.float32, seed=seed2
    )
    element = tf.cond(
        pred=tf.less(coin_toss, tf.cast(self.apply_prob, tf.float32)),
        true_fn=lambda: blurred,
        false_fn=lambda: element,
    )
    return element


class RandomFlipLeftRightVideo(ElementWiseRandomTransform):
  """Randomly flips an all frames in input video.

  For an input of shape (B,H,W,C), this transformation randomly
  flips all elements in batch B horizontally with 50% probability
  of being flipped.
  """

  @typechecked
  def random_map_element(
      self, element: TfArray["B H W C"], seed: tf.Tensor
  ) -> TfArray["B H W C"]:
    flip_mask = tf.random.stateless_uniform(shape=(1, 1, 1, 1), seed=seed) < 0.5

    # Apply the mask to the batch
    flipped_images = tf.where(
        flip_mask, tf.image.flip_left_right(element), element
    )
    return flipped_images


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Resize(base.ElementWiseTransform):
  """Resize images and corresponding segmentations, etc.

  By default uses resize method "area" for float inputs and resize method
  "nearest" for int inputs.

  Attributes:
    height: Output height of the image(s).
    width: Output width of the image(s).
    method: The resizing method to use. Defaults to "AUTO" in which case the
      resize method is either "area" (for float inputs) or "nearest" (for int
      inputs). Other possible choices are "bilinear", "lanczos3", "lanczos5",
      "bicubic", "gaussian", "nearest", "area", or "mitchellcubic". See
      `tf.image.resize` for details.
  """

  height: int
  width: int
  method: str = "AUTO"

  @typechecked
  def map_element(self, element: TfArray["*b H W C"]) -> TfArray["*b H2 W2 C"]:
    # Determine resize method based on dtype (e.g. segmentations are int).
    method = self.method
    if method == "AUTO":
      method = "nearest" if element.dtype.is_integer else "area"

    batch_dims = tf.shape(element)[:-3]
    flat_imgs = einops.rearrange(element, "... h w c -> (...) h w c")

    resized_imgs = tf.image.resize(
        flat_imgs, (self.height, self.width), method=method
    )
    return tf.reshape(
        resized_imgs,
        tf.concat([batch_dims, tf.shape(resized_imgs)[-3:]], axis=0),
    )
