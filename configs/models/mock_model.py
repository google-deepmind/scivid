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

"""Mock model for fast prototyping & debugging."""

from kauldron import konfig


# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  from scivid.models.backbones import pass_through
  from scivid.data.transforms import tf_transforms
  from scivid.data.transforms import transforms
# pylint: enable=g-import-not-at-top

# Notations:
#   B: batch size
#   T: temporal dimension (number of frames)
#   K: backbone-dependent sequence dimension of the frame-level features
#      (typically H*W)
#   H: height dimension
#   W: width dimension
#   D: embedding dimension

# Kauldron key for features to evaluate.
# These features are expected to be of shape (B, T, K, D). Here, K=1.
# Used by eg. classification, pressure forecasting, tracking.
FEATURES: str = "preds.mean_pixel"


# Kauldron key for spatiotemporally dense features to evaluate.
# These grid features are expected to be of shape (B, T, H, W, D).
# Used by eg. frame forecasting.
GRID_FEATURES: str = "preds.image"

# Expected image size for the model (typical pretraining resolution)
IMAGE_SIZE: int = 224
# Expected number of frames for the model (typical pretraining duration)
NUM_FRAMES: int = 16

# For pixel-wise evals. Upsampling factor from GRID_FEATURES tensor.
PATCH_SIZE: tuple[int, int, int] = (
    1,
    14,
    14,
)


def get_config(
    cfg: kd.train.Trainer,
) -> kd.train.Trainer:
  """The default hyperparameter configuration for the mock model."""
  cfg.aux.update({
      # "Readout" refers to the lightweight module learned to map backbones
      # features to task outputs.
      "readout": {
          # Kauldron key for the backbone input.
          "model_inputs_key": "image",
          "model": pass_through.PassThroughModel(patch_size=PATCH_SIZE),
          # Connecting the model output to the readout inputs with Kauldron keys
          "readout_inputs": {
              "grid_features": GRID_FEATURES,
              "features": FEATURES,
          },
          "model_name": "mock_model",
          "readout_heads": {"reduction_axes": (1, 2)},
      },
      "patch_size": PATCH_SIZE,
      "custom_transform": [
          # For each model, we add custom transforms to resample the input to
          # the expected number of frames and expected spatial resolution.
          tf_transforms.Resize(
              key="video",
              height=IMAGE_SIZE,
              width=IMAGE_SIZE,
          ),
          transforms.RepeatFrames(
              key="video",
              divisible_by=NUM_FRAMES,
          ),
      ],
  })

  cfg.rng_streams = kd.train.RngStreams([
      kd.train.RngStream("default", train=True, eval=True),
  ])

  return cfg
