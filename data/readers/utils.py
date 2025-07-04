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

"""Utilities for processing videos in Kauldron."""

from collections.abc import Callable
import enum
from typing import Any, Optional, Sequence
import tensorflow as tf


class Sampling(enum.StrEnum):
  RANDOM_CLIP = 'random_clip'
  MULTI_CLIP = 'multi_clip'
  CLIP = 'clip'
  MIDDLE_CLIP = 'middle_clip'
  LINSPACE_CLIP = 'linspace_clip'


class ResizeMethod(enum.StrEnum):
  CROP = 'crop'
  INTERPOLATE_WITH_ASPECT_RATIO = 'interpolate_ar'
  INTERPOLATE_WITHOUT_ASPECT_RATIO = 'interpolate_no_ar'
  INTERPOLATE_WITH_ASPECT_RATIO_AND_CROP = 'interpolate_ar_crop'
  INTERPOLATE_WITHOUT_ASPECT_RATIO_AND_CROP = 'interpolate_no_ar_crop'


def process_video(
    sequence: dict[str, Any],
    *,
    input_name: str,
    output_name: str,
    frame_idx: tf.Tensor,
    num_frames: Optional[int],
    im_size: tuple[int, int],
    im_channels: int = 3,
    num_clips: int = 1,
    resize_method: ResizeMethod = ResizeMethod.INTERPOLATE_WITHOUT_ASPECT_RATIO,
    interpolation_method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
    decode_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    dtype: tf.DType = tf.uint8,
    flip_channels: bool = False,
    grayscale_to_rgb: bool = False,
) -> dict[str, tf.Tensor]:
  """Processes a sequence input to return an output video.

  Responsible for subsampling, decoding, resizing.
  Note: when adding additional arguments for further processing,
  consider whether the desired behavior could not be achieved by a standalone
  kauldron data transform (as part of the transforms passed to the reader).

  Args:
    sequence: input sequence example
    input_name: field name for the input to process
    output_name: field name to use for the output (processed video)
    frame_idx: tensor holding the frame indices to gather from the input video
      (typically used to subsample the video). See process_idx for examples
    num_frames: expected number of frames of the output video (should be
      consistent with frame_idx)
    im_size: a tuple (image, width) specifying the target spatial resolution
    im_channels: expected number of channels of the output video
    num_clips: expected number of clips concatenated to form the output video
      (should be consistent with frame_idx)
    resize_method: ResizeMethod to use to obtain a video of the target spatial
      resolution - see ResizeMethod for the possible values. By default,
      interpolates withpout preserving the aspect ratio.
    interpolation_method: the tf.image.ResizeMethod used by tf.image.resize to
      rescale to the target resolution (if resize_method is one of
      ResizeMethod.INTERPOLATE_*)
    decode_fn: optional function used to decode the sequence. By default,
      tf.io.decode_image is used (with expand_animations=False)
    dtype: desired type after decoding the sequence
    flip_channels: whether or not to reverse the order of the channels; set to
      False by default. Useful when frames are stored in BGRformat
    grayscale_to_rgb: whether to convert the sequence input from grayscale to
      rgb, when expected number of channels is 3

  Returns:
    A dictionary of the form {output_name: preprocessed_video}.
  """

  raw_imgs = sequence[input_name]

  raw_imgs = tf.gather(raw_imgs, frame_idx)

  if not decode_fn:
    decode_fn = lambda x: tf.io.decode_image(
        x, expand_animations=False, dtype=dtype
    )
  images = tf.map_fn(decode_fn, raw_imgs, dtype=dtype)

  if flip_channels:
    images = tf.reverse(images, axis=[-1])

  if im_size is not None:
    if resize_method == ResizeMethod.CROP:
      resize = False
      preserve_aspect_ratio = False
      crop = True
    elif resize_method == ResizeMethod.INTERPOLATE_WITH_ASPECT_RATIO:
      resize = True
      preserve_aspect_ratio = True
      crop = False
    elif resize_method == ResizeMethod.INTERPOLATE_WITHOUT_ASPECT_RATIO:
      resize = True
      preserve_aspect_ratio = False
      crop = False
    elif resize_method == ResizeMethod.INTERPOLATE_WITH_ASPECT_RATIO_AND_CROP:
      resize = True
      preserve_aspect_ratio = True
      crop = True
    elif (
        resize_method == ResizeMethod.INTERPOLATE_WITHOUT_ASPECT_RATIO_AND_CROP
    ):
      resize = True
      preserve_aspect_ratio = False
      crop = True
    else:
      raise ValueError(f'Unknown resize method: {resize_method}')

    images = resize_and_crop_image(
        images,
        im_size,
        resize=resize,
        preserve_aspect_ratio=preserve_aspect_ratio,
        crop=crop,
        interpolation_method=interpolation_method,
    )

  if grayscale_to_rgb and im_channels == 3:
    # This fix is needed because while in tf.data pipeline set_shape is not
    # failing when called on the output of the wrong shape it does so in pygrain
    # pipeline. Hence if we decode images as grayscale we need to repeat the
    # channel dimension 3 times to match the expected shape.
    images = tf.image.grayscale_to_rgb(images)

  if num_frames is not None:
    images.set_shape(
        [num_clips * num_frames, images.shape[1], images.shape[2], im_channels]
    )
  else:
    images.set_shape(
        [images.shape[0], images.shape[1], images.shape[2], im_channels]
    )

  return {output_name: images}


def resize_and_crop_image(
    images: tf.Tensor,
    im_size: tuple[int, int] = (256, 256),
    resize: bool = False,
    preserve_aspect_ratio: bool = False,
    crop: bool = False,
    interpolation_method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
) -> tf.Tensor:
  """Resizes and crops an image according to im_size.

  Resizes shorter size to im_size h or w and then takes centre crop.
  The ratio is determined by calculating the ratio of the shorter side to the
  im_size.

  Args:
    images (tf.Tensor): input batch of images (N,H,W,C).
    im_size (tuple[int, int]): tuple of images (height, width) for output crop
      size.
    resize (bool): whether to resize the image or not.
    preserve_aspect_ratio (bool): whether to preserve aspect ratio when
      resizing.
    crop (bool): whether to crop the image or not.
    interpolation_method (tf.image.ResizeMethod): interpolation method for
      resizing.

  Returns:
    Tf.Tensor: cropped and resized tensor of batch of images
      (N, im_size[0], im_size[1], C).
  """
  if resize:
    if not preserve_aspect_ratio:
      h = im_size[0]
      w = im_size[1]
    else:
      new_ratio = tf.cond(
          tf.less_equal(tf.shape(images)[1], tf.shape(images)[2]),
          lambda: tf.shape(images)[1] / im_size[0],
          lambda: tf.shape(images)[2] / im_size[1],
      )
      shape = tf.cast(tf.shape(images), tf.float64)
      h = tf.cast(shape[1] / new_ratio, tf.int32)
      w = tf.cast(shape[2] / new_ratio, tf.int32)

    images = tf.image.resize(images, (h, w), method=interpolation_method)

  else:
    h = im_size[0]
    w = images.shape[2]

  if crop:
    offset_h = int((h - im_size[0]) / 2)
    offset_w = int((w - im_size[1]) / 2)
    images = tf.image.crop_to_bounding_box(
        images, offset_h, offset_w, im_size[0], im_size[1]
    )

  return images


def process_idx(
    vid_len: Optional[int] = None,
    *,
    output_name: str,
    num_frames: Optional[int],
    stride: int,
    sampling: str | Sampling,
    num_clips: int,
    max_start_frame: int | None = None,
) -> dict[str, tf.Tensor]:
  """Returns the frame indices to extract from the video.

  Args:
    vid_len: length of the sequence of frames to extract clips from
    output_name: field name to use for the output (processed video)
    num_frames: number of frames to extract - per clip - from the sequence of
      frames. If none, requires sampling to be 'clip' and returns the full video
      (possibly subsampled corresponding to the stride value).
    stride: Stride for sampling frames (ie subsampling factor).
    sampling: Sampling method to use, one of:
      * clip: Sample a single clip from the video starting at frame 0.
      * random_clip: Sample a random clip from the video with random start
        frame.
      * multi_clip: Sample multiple clips from the video, number of clips is
        defined by num_clips.
      * middle_clip: Sample the middle clip with fixed stride eg. for
        single-clip fast eval.
      * linspace_clip: Sample a clip from the video starting at frame 0 and
        ending at the last frame (using adaptive stride).
    num_clips: Number of clips to extract, only used for 'multi_clip' sampling
    max_start_frame: When 'random_clip' sampling is used, and if not None, clip
      start frame is sampled uniformly from the first max_start_frame frames.

  Returns:
    A dictionary holding the frame indices and optionally the padding mask.
  """
  if max_start_frame is not None:
    if sampling != 'random_clip':
      raise ValueError(
          'max_start_frame is only supported with random_clip sampling, but'
          f' got max_start_frame={max_start_frame} and'
          f' sampling={sampling}'
      )
    # Limit the sequence length so that the beginning of the clip is sampled
    # uniformly from the first valid max_start_frame frames, ensuring that
    # that num_frames can be sampled without exceeding the video length.
    vid_len = tf.minimum(vid_len, num_frames + max_start_frame)

  frame_dict = {}
  if num_frames is None:
    if sampling != Sampling.CLIP:
      raise ValueError(
          'if num_frames is None sampling mode must be "clip" to return full'
          ' video'
      )
    start_frame = 0
    frame_idx = tf.range(start_frame, vid_len, stride)

  else:
    #  sample k frames with s stride from random idx
    if sampling == Sampling.RANDOM_CLIP:
      end_frame = vid_len - (num_frames * stride)
      if end_frame > 0:
        start_frame = tf.random.uniform(
            shape=(),
            minval=0,
            maxval=end_frame,
            dtype=tf.int32,
        )
      else:
        start_frame = 0
      frame_idx = tf.range(
          start_frame, start_frame + num_frames * stride, stride
      )

    #  sample k frames with s stride from idx 0
    elif sampling == Sampling.CLIP:
      end_frame = num_frames * stride
      start_frame = 0
      frame_idx = tf.range(start_frame, end_frame, stride)

    elif sampling == Sampling.MIDDLE_CLIP:
      # sample the middle clip with fixed stride for single-clip fast eval.
      clip_dur = num_frames * stride
      diff = tf.maximum(0, vid_len - clip_dur)
      start_frame = diff // 2
      end_frame = vid_len - diff // 2
      frame_idx = tf.cast(
          tf.linspace(start_frame, end_frame, num_frames), dtype=tf.int32
      )

    elif (
        sampling == Sampling.MULTI_CLIP
    ):  # sampling n clips of k frames by calculating stride required to
      # cover entire video
      if num_clips < 2:
        raise ValueError(f'num_clips must be > 1: {num_clips}')
      max_offset = tf.maximum(0, vid_len - stride * num_frames)
      offsets = tf.linspace(0.0, tf.cast(max_offset, tf.float32), num_clips)
      offsets = tf.cast(offsets, tf.int32)

      frame_idx = tf.range(0, stride * num_frames, delta=stride)[None, :]
      frame_idx += offsets[:, None]
      frame_idx = tf.reshape(frame_idx, [num_frames * num_clips])

    elif sampling == Sampling.LINSPACE_CLIP:
      frame_idx = tf.cast(
          tf.math.round(tf.linspace(0, vid_len - 1, num_frames)),
          tf.int32,
      )
    else:
      raise ValueError(f'Unknown sampling: {sampling}')

    # anything after last frame is padded with last frame
    frame_idx = tf.where(
        tf.greater(frame_idx, vid_len - 1),
        vid_len - 1,
        frame_idx,
    )
    frame_idx.set_shape([num_frames * num_clips])

  frame_dict[output_name] = frame_idx

  return frame_dict


def process_uncompressed_tensor(
    sequence: dict[str, Any],
    *,
    input_name: str,
    output_name: str,
    frame_idx: tf.Tensor,
    num_frames: int,
    num_clips: int,
    tensor_shape: Sequence[Optional[int]],
    dtype: tf.DType = tf.float32,
    decoding_type: str = 'parse_tensor',
) -> dict[str, tf.Tensor]:
  """Processes uncompressed tf tensor data."""
  if decoding_type == 'parse_tensor':
    decode_fn = lambda x: tf.io.parse_tensor(x, out_type=dtype)
  elif decoding_type == 'decode_raw':
    decode_fn = lambda x: tf.io.decode_raw(x, out_type=dtype)
  else:
    raise ValueError(f'Unknown decoding type: {decoding_type}')
  tensor_bytes = tf.gather(sequence[input_name], frame_idx)
  tensor = tf.map_fn(decode_fn, tensor_bytes, dtype=dtype)
  if num_frames is None or num_clips is None:
    tensor.set_shape([tensor.shape[0], *tensor_shape])
  else:
    tensor.set_shape([num_clips * num_frames, *tensor_shape])
  return {output_name: tensor}
