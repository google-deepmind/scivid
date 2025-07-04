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

"""HuggingFace VideoMAE v1 model."""

from kauldron import konfig


# pylint: disable=g-import-not-at-top
with konfig.imports():
  import jax
  from kauldron import kd
  from scivid.data.transforms import transforms as data_transforms
  from scivid.models.backbones import videomae
  import transformers
  import torch
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
# These features are expected to be of shape (B, T, K, D). Here, K=H*W.
# Used by eg. classification, pressure forecasting, tracking.
FEATURES: str = "preds.features"


# Kauldron key for spatiotemporally dense features to evaluate.
# These grid features are expected to be of shape (B, T, H, W, D).
# Used by eg. frame forecasting.
GRID_FEATURES: str = "preds.grid_features"

# Refer to https://huggingface.co/docs/transformers/main/model_doc/videomae
# for other model name options.
MODEL_NAME: str = "MCG-NJU/videomae-base"

# Expected image size for the model (set to the pretraining spatial resolution)
IMAGE_SIZE: int = 224
# Expected number of frames for the model (set to the pretraining clip duration)
NUM_FRAMES: int = 16

# Spatial and temporal patch size for the image encoder.
PATCH_SIZE: int = 16
TUBELET_SIZE: int = 2

# Latent dimension of the last hidden layer.
LATENT_DIM: int = 768


def get_config(
    cfg: kd.train.Trainer,
) -> kd.train.Trainer:
  """The default hyperparameter configuration for the mock model."""
  cfg.aux.update({
      # "Readout" refers to the lightweight module learned to map backbone
      # features to task outputs.
      "readout": {
          # Kauldron key for the backbone input.
          "model_inputs_key": "image",
          "model": videomae.HfModelWrapper(
              model=transformers.VideoMAEModel.from_pretrained(MODEL_NAME),
              model_output_dims=(
                  int(NUM_FRAMES / TUBELET_SIZE),
                  int(IMAGE_SIZE / PATCH_SIZE),
                  int(IMAGE_SIZE / PATCH_SIZE),
                  LATENT_DIM,
              ),
              torch_device=(
                  torch.device("cuda")
                  if torch.cuda.is_available()
                  else torch.device("cpu")
              ),
              jax_device=konfig.resolve(jax.devices())[0].platform,
          ),
          # Connecting the model output to the readout inputs with Kauldron keys
          "readout_inputs": {
              "grid_features": GRID_FEATURES,
              "features": FEATURES,
          },
          "model_name": "hf_videomae",
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
          # HfProcess expects the input to be in the format T*C*H*W
          # cf https://huggingface.co/docs/transformers/model_doc/videomae
          # #transformers.VideoMAEForPreTraining.forward.example.
          kd.data.Rearrange(key="video", pattern="t h w c -> t c h w"),
          # The HfPreprocess transform applies the image processor expected
          # by the specified model from raw pixel values (in [0, 1] range)
          # to the model's expected input space.
          # Here we are using the default preprocessor config for the model
          # https://huggingface.co/MCG-NJU/videomae-base/blob/main/preprocessor_config.json;
          # disabling certain options for consistency across models and evals.
          data_transforms.HfPreprocess(
              key="video",
              processor=transformers.AutoImageProcessor.from_pretrained(
                  MODEL_NAME,
                  # We disable default rescaling by 1/255, as each eval scales
                  # video inputs to the appropriate range.
                  do_rescale=False,
                  # We resize spatially outside of the HfPreprocess transform,
                  # as the spatial resizing fails if the values are not strictly
                  # in the [0, 1] range.
                  do_resize=False,
              ),
          ),
          # We rearrange again in order to match the expected image
          # input to the model that is standard across backbones.
          kd.data.Rearrange(key="video", pattern="t c h w -> t h w c"),
      ],
  })

  cfg.rng_streams = kd.train.RngStreams([
      kd.train.RngStream("default", train=True, eval=True),
  ])

  return cfg
