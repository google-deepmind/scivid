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

"""4DS-B-dist-e Backbone (ViT-B) from Scaling 4D Representations.

See paper: https://arxiv.org/abs/2412.15212
github repo: https://github.com/google-deepmind/representations4d
"""

import os
from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  from scivid.data.transforms import transforms as data_transforms
  from scivid.models.backbones import scaling4d
# pylint: enable=g-import-not-at-top

# Kauldron key for aggregated features.
FEATURES: str = "preds.features"


# Kauldron key for spatiotemporally dense grid features.
GRID_FEATURES: str = "preds.grid_features"

# Path to weights for 4DS-B-dist-e model (88M params).
# Path to the pretrained model weights.
# The weights can be downloaded from:
# https://storage.googleapis.com/representations4d/checkpoints/scaling4d_dist_b.npz  # pylint: disable=line-too-long
PRETRAINED_WEIGHTS_PATH = os.getenv('SCALING4D_CHECKPOINT_PATH')
if PRETRAINED_WEIGHTS_PATH is None:
  raise ValueError(
      "Please download the scaling4d checkpoint weights stored at"
      " https://storage.googleapis.com/representations4d/checkpoints/scaling4d_dist_b.npz"
      " and provide path to the downloaded scaling4d checkpoint weights using"
      " the SCALING4D_CHECKPOINT_PATH environment variable (see README.md for"
      " details)."
  )

# Expected image size for the model (set to the pretraining spatial resolution)
IMAGE_SIZE: int = 224
# Expected number of frames for the model (set to the pretraining clip duration)
NUM_FRAMES: int = 16

# Spatial and temporal patch size for the image encoder.
MODEL_PATCH_SIZE: tuple[int, int, int] = (2, 16, 16)


def get_config(
    cfg: kd.train.Trainer,
) -> kd.train.Trainer:
  """The default hyperparameter configuration for the mock model."""
  cfg.init_transform = scaling4d.Loader(
      checkpoint_path=PRETRAINED_WEIGHTS_PATH, backbone_prefix="model"
  )

  cfg.aux.update({
      # "Readout" refers to the lightweight module learned to map backbone
      # features to task outputs.
      "readout": {
          # Kauldron key for the backbone input.
          "model_inputs_key": "image",
          "model": scaling4d.Scaling4DModel(
              image_size=(IMAGE_SIZE, IMAGE_SIZE),
              num_input_frames=NUM_FRAMES,
              model_patch_size=MODEL_PATCH_SIZE,
          ),
          # Connecting the model output to the readout inputs with Kauldron keys
          "readout_inputs": {
              "grid_features": GRID_FEATURES,
              "features": FEATURES,
          },
          "model_name": "scaling4d",
          "readout_heads": {"reduction_axes": (1, 2)},
      },
      "custom_transform": [
          # For each model, we add custom transforms to resample the input to
          # the expected number of frames and expected spatial resolution
          # (following the spatiotemporal resolution used during pretraining).
          data_transforms.RepeatFrames(
              key="video",
              divisible_by=NUM_FRAMES,
          ),
          kd.data.Resize(key="video", size=(IMAGE_SIZE, IMAGE_SIZE)),
      ],
  })

  cfg.rng_streams = kd.train.RngStreams([
      kd.train.RngStream("params", train=True, eval=True),
      kd.train.RngStream("default", train=True, eval=True),
  ])

  return cfg
