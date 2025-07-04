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

"""Evaluation config for Weatherbench future frame prediction."""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  from scivid.data.transforms import transforms as data_transforms
  from scivid.models import readouts
  from scivid.data.readers import weather
  from scivid.metrics import regression
  from scivid.models import readout_wrapper


# pylint: enable=g-import-not-at-top
def _make_ds(
    subset: str = "train",
    batch_size: int = 1,
    num_epochs: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    custom_transforms=None,
    ensure_full_dataset_eval: bool = False,
):
  """Returns a dataset for training or evaluation."""
  if subset not in ["train", "online_dev", "offline_dev", "test"]:
    raise ValueError(f"Unknown subset: {subset}.")

  if ensure_full_dataset_eval:
    # Fail if we haven't set a batch_size that allows to evaluate on the full
    # dataset.
    if (subset == "test") and (732 % batch_size != 0):
      raise ValueError(
          f"batch_size {batch_size} must be a divider of 732 for the eval to"
          " run on the whole test set."
      )
    if (subset == "offline_dev") and (730 % batch_size != 0):
      raise ValueError(
          f"batch_size {batch_size} must be a divider of 730 for the eval to"
          " run on the whole offline_dev set."
      )
  transforms = [
      # Prepare a preprocessed input to the model.
      kd.data.Elements(copy={"image": "model_input"}),
      data_transforms.Normalize(
          key="model_input",
          in_vrange=(0.0, 1.0),
          normalize_mean=weather.FILTERED_MEAN,
          normalize_std=weather.FILTERED_STD,
      ),
  ]
  # Apply model-specific transforms to the input.
  # Note: depending on the model, we may be applying several normalization steps
  # or several resampling operations in a row (eg. to 224x224 then to 288x288).
  # Current configurations ensure consistency across backbones for a given eval;
  # but for optimal performance of a model x eval configuration, please check
  # the preprocessing steps carefully.
  if custom_transforms is not None:
    for transform in custom_transforms:
      transform.key = "model_input"
    transforms += custom_transforms
  return weather.PyGrainEra5Reader(
      subset=subset,
      input_steps=weather.INPUT_STEPS,
      target_steps=weather.TARGET_STEPS,
      timestep_hours=weather.TIMESTEP_HOURS,
      batch_size=batch_size,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      transforms=transforms,
  )


def get_config(cfg: kd.train.Trainer) -> kd.train.Trainer:
  """The default hyperparameter configuration."""

  # Model.
  pred_timestep = weather.TARGET_STEPS
  num_channels = weather.NUM_FILTERED
  img_dimensions = (weather.IMAGE_HEIGHT, weather.IMAGE_WIDTH)
  readout_model = readout_wrapper.ReadoutWrapper(
      # Inner model
      model_inputs={
          cfg.aux["readout"]["model_inputs_key"]: "batch.model_input",
      },
      model=cfg.aux["readout"]["model"],
      # Readout heads
      readout_inputs={
          "future_rgb": {
              # Readout trained to predict a residual to the last input image.
              "x": "batch.image",
              "y": cfg.aux["readout"]["readout_inputs"][
                  "grid_features"
              ],  # default to a field of preds/
          },
          "eulerian_persistence": {
              "x": "batch.image",
          },
      },
      readout_heads={
          "future_rgb": readouts.UnnormalizePredAndAddInput(
              inner_readout=readouts.DPTReadout(
                  output_resolution=img_dimensions,
                  pred_seq_len=pred_timestep,
                  output_channels=num_channels,
                  dpt_feature_dims=(1024, 1024, 1024, 1024),
                  dpt_intermediate_channels=512,
              ),
              residual_mean=weather.FILTERED_RESIDUAL_MEAN,
              residual_std=weather.FILTERED_RESIDUAL_STD,
          ),
          # Control baseline that copies the last input frame at each timestep.
          "eulerian_persistence": readouts.EulerianPersistence(
              num_output_frames=pred_timestep
          ),
      },
      finetune=cfg.ref.aux["finetune"],
  )
  cfg.model = readout_model

  # Readout heads do not assume a particular number of frames as input.
  # (no need to resize output of the model over temporal axis)
  cfg.aux["task_num_frames"] = weather.TARGET_STEPS

  # We set the default depth to 0.1 as we have observed that using features from
  # the early layers tends to perform better on weatherbench.
  cfg.aux["readout_depth_fraction"] = (0.1,)

  # Training losses and optimization hyperparameters.
  cfg.train_losses = {
      "dpt.weighted_L1": regression.Weatherbench2AreaWeightedL1(
          pred="preds.readouts.future_rgb",
          target="batch.future",
          residual_std=weather.FILTERED_RESIDUAL_STD,
      ),
  }

  # Add gradient clipping as DPT readout is unstable with learning rates > 1e-3.
  cfg.aux["grad_clip"] = 1.0

  # Shorter training to prevent overfitting.
  cfg.num_train_steps = 10000

  # Metrics.
  cfg.train_metrics = {
      "dpt.PSNR": kd.metrics.Psnr(
          pred="preds.readouts.future_rgb", target="batch.future"
      ),
      "dpt.wRMSE": regression.Weatherbench2AreaWeightedRMSE(
          pred="preds.readouts.future_rgb", target="batch.future"
      ),
  }

  eval_metrics = {
      "dpt.PSNR": kd.metrics.Psnr(
          pred="preds.readouts.future_rgb", target="batch.future"
      ),
      "dpt.wRMSE": regression.Weatherbench2AreaWeightedRMSE(
          pred="preds.readouts.future_rgb", target="batch.future"
      ),
      # This baseline is expected to get a normalized_score of exactly 0. for
      # the offline dev setting (final_eval_dev); and -0.0024 for the online dev
      # setting (eval) (because we compute copy and graphcast rescaling values
      # for this score in the offline dev setting.)
      "control.wRMSE": regression.Weatherbench2AreaWeightedRMSE(
          pred="preds.readouts.eulerian_persistence", target="batch.future"
      ),
  }

  # Training dataset.
  # custom_transform is used to apply model-specific preprocessing steps
  # eg. resizing for videoprism or normalization for imagenet_dinov2
  custom_transforms = None
  if "custom_transform" in cfg.aux:
    custom_transforms = cfg.aux["custom_transform"]

  cfg.train_ds = _make_ds(
      subset="train",
      batch_size=16,
      shuffle=True,
      seed=42,
      custom_transforms=custom_transforms,
  )

  # Evaluations.
  # Set to True below to run the final eval on the test set.
  cfg.aux["run_eval_on_test"] = False

  @konfig.ref_fn
  def get_evals(aux) -> dict[str, kd.evals.Evaluator]:
    """Returns a dictionary of evaluation configurations."""
    evals = {
        "minival": kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(aux.eval_interval_steps),
            num_batches=None,
            ds=_make_ds(
                subset="online_dev",  # 78 samples
                num_epochs=1,
                batch_size=8,  # dropping last incomplete batch (=> 72 samples)
                shuffle=False,
                seed=0,
                custom_transforms=custom_transforms,
            ),
            metrics=eval_metrics,
        ),
        "val": kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(aux.long_eval_interval_steps),
            num_batches=None,
            ds=_make_ds(
                subset="offline_dev",  # 730 samples
                num_epochs=1,
                # batch_size should be a divider of 730 to ensure no samples
                # are dropped.
                batch_size=10,
                shuffle=False,
                seed=0,
                custom_transforms=custom_transforms,
                ensure_full_dataset_eval=True,
            ),
            metrics=eval_metrics,
        ),
    }
    if aux.run_eval_on_test:
      evals["test"] = kd.evals.Evaluator(
          run=kd.evals.Once(step=cfg.ref.num_train_steps),
          num_batches=None,
          ds=_make_ds(
              subset="test",  # 732 samples
              num_epochs=1,
              # batch_size should be a divider of 732(=4*3*61) to ensure no
              # samples are dropped.
              batch_size=12,
              shuffle=False,
              seed=0,
              custom_transforms=custom_transforms,
              ensure_full_dataset_eval=True,
          ),
          metrics=eval_metrics,
      )
    return evals

  cfg.evals = get_evals(cfg.ref.aux)

  return cfg
