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

"""Evaluation config for Typhoon future pressure forecasting."""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  from scivid.models import readouts
  from scivid.models import readout_wrapper
  from scivid.data.readers import typhoon
  from scivid.metrics import regression as regression_metrics
  from scivid.summaries import images
  from scivid.data.transforms import transforms as scivid_transforms
# pylint: enable=g-import-not-at-top

# Range of the digital typhoon infrared values.
_INFRARED_MIN = 170  # https://arxiv.org/pdf/2311.02665 See Appendix C.3 pg 19
_INFRARED_MAX = 300


def _make_ds(
    subset: str,
    num_epochs: int | None = None,
    custom_transforms=None,
    batch_size: int = 16,
    num_frames: int = typhoon.NUM_INPUT_FRAMES,
    pred_len: int = typhoon.PRED_LEN,
    im_size: tuple[int, int] = (
        typhoon.IMAGE_RESOLUTION,
        typhoon.IMAGE_RESOLUTION,
    ),
    shuffle: bool = False,
    sampling: str = "clip",
    max_start_frame: int | None = None,
    debug: bool = False,
):
  """Returns a dataset for training or evaluation.

  Args:
    subset: Which subset of the dataset to use, one of "train" / "val" / "test"
      / "train_val".
    num_epochs: The number of epochs to repeat the dataset. If None, repeats the
      dataset indefinitely.
    custom_transforms: A list of custom data transforms specific to the model
      being evaluated.
    batch_size: Number of samples in a batch.
    num_frames: Number of input frames to extract (may be further resampled by
      the model's custom transforms to match the pretraining clip duration).
    pred_len: Number of future time steps to forecast.
    im_size: Spatial resolution of the clip (may be further resized by
      the model's custom transforms to match the pretraining resolution).
    shuffle: Whether to shuffle the dataset.
    sampling: Sampling method to use, one of:
      * 'clip': Sample a single clip from the video starting at frame 0.
      * 'random_clip': Sample a random clip from the video with random start
        frame.
      * see PygrainVideoReader for additional values (not used here).
    max_start_frame: When 'random_clip' sampling is used, and if not None, clip
      start frame is sampled uniformly from the first max_start_frame frames.
    debug: If True, load additional fields in datasets for debugging purposes.
  """

  @konfig.ref_fn
  def get_fields_transform(debug: bool):
    # Prepare field for repeating last pressure value pred_len times for
    # copy-last-value baseline.
    rename_keys = {
        "last_pressure": "repeated_last_pressure",
    }
    # Fields which are necessary to keep for training.
    keep_keys = [
        "pressure",
        "normalized_pressure",
        "global_mean_pressure",
        "video_id",
        "mean_future_pressures",
    ]
    if debug:
      # Keep a copy of "video" untouched for debugging purposes, and apply all
      # input preprocessing transforms to "model_input".
      keep_keys.append("video")
      copy_keys = {
          "video": "model_input",
      }
    else:
      # Rename "video" to "model_input". Image transforms will be applied to
      # "model_input" field and no "video" field will be present, preventing
      # extra memory usage.
      rename_keys["video"] = "model_input"
      copy_keys = {}

    return kd.data.Elements(
        keep=keep_keys, copy=copy_keys, rename=rename_keys
    )

  transforms = [
      get_fields_transform(debug=debug),
      # Normalize input and target images following
      # https://github.com/kitamoto-lab/benchmarks/blob/1bdbefd7c570cb1bdbdf9e09f9b63f7c22bbdb27/forecasting/Dataloader/SequenceDatamodule.py#L112
      # Value range normalization is applied BEFORE the model custom transformsm
      # which potentially assume that image is normalized to [0, 1] range.
      kd.data.ValueRange(
          key=["model_input"],
          vrange=(0, 1),
          in_vrange=(_INFRARED_MIN, _INFRARED_MAX),
          clip_values=True,
      ),
  ]

  if custom_transforms is not None:
    for transform in custom_transforms:
      transform.key = "model_input"
    transforms += custom_transforms

  transforms += [
      # Repeat pressure value from last input step for copy-last-value baseline.
      scivid_transforms.Repeat(
          key="repeated_last_pressure",
          pattern=" -> t",
          axes_lengths={"t": pred_len},
      ),
  ]
  return typhoon.TyphoonPygrainReader(
      subset=subset,
      batch_size=batch_size,
      # Only shuffle train and trainval datasets.
      shuffle=shuffle,
      sampling=sampling,
      num_epochs=num_epochs,
      max_start_frame=max_start_frame,
      transforms=transforms,
      load_pressure=True,
      im_size=im_size,
      num_frames=num_frames,
      pred_len=pred_len,
  )


def get_config(cfg: kd.train.Trainer) -> kd.train.Trainer:
  """The default hyperparameter configuration."""

  # Model.
  pred_len = typhoon.PRED_LEN
  mean_offset_key = "batch.global_mean_pressure"
  readout_model = readout_wrapper.ReadoutWrapper(
      # Inner model
      model_inputs={
          cfg.aux["readout"]["model_inputs_key"]: "batch.model_input",
      },
      model=cfg.aux["readout"]["model"],
      # Readout heads
      readout_inputs={
          "linear_future_pressure": {
              "inputs": cfg.aux["readout"]["readout_inputs"]["features"],
              "output_offsets": mean_offset_key,
          },
          "attn_future_pressure": {
              "inputs": cfg.aux["readout"]["readout_inputs"]["features"],
              "output_offsets": mean_offset_key,
          },
      },
      readout_heads={
          "linear_future_pressure": readouts.ShiftPred(
              inner_readout=readouts.LinearReadout(
                  num_classes=pred_len, reduce_axis=(-2, -3)
              ),
          ),
          "attn_future_pressure": readouts.ShiftPred(
              readouts.AttentionReadout(
                  num_classes=pred_len, num_params=1024, num_heads=16
              )
          ),
      },
      finetune=cfg.ref.aux["finetune"],
  )

  cfg.model = readout_model

  # Readout heads do not assume a particular number of frames as input.
  # (no need to resize output of the model over temporal axis)
  cfg.aux["task_num_frames"] = None

  # Training losses and optimization hyperparameters.
  cfg.train_losses = {
      "L2_loss_linear_head": kd.losses.L2(
          preds="preds.readouts.linear_future_pressure.unshifted_pred",
          targets="batch.normalized_pressure",
      ),
      "L2_loss_attn_head": kd.losses.L2(
          preds="preds.readouts.attn_future_pressure.unshifted_pred",
          targets="batch.normalized_pressure",
      ),
  }

  # Add gradient clipping to address instability we sometimes observe when using
  # videomae-base backbone.
  cfg.aux["grad_clip"] = 1.0

  # Metrics.
  cfg.train_metrics = {  # https://arxiv.org/pdf/2311.02665 pg 10, table 5
      "linear.typhoon_scores": (
          regression_metrics.TyphoonTemporalRegressionMetrics(
              preds="preds.readouts.linear_future_pressure.shifted_pred",
              targets="batch.pressure",
          )
      ),
      "attention.typhoon_scores": (
          regression_metrics.TyphoonTemporalRegressionMetrics(
              preds="preds.readouts.attn_future_pressure.shifted_pred",
              targets="batch.pressure",
          )
      ),
      # Copy-last-input-pressure oracle.
      "oracle.typhoon_scores": (
          regression_metrics.TyphoonTemporalRegressionMetrics(
              preds="batch.repeated_last_pressure",
              targets="batch.pressure",
          )
      ),
      # Control baseline, predicting mean pressure values across train samples.
      "control.typhoon_scores": (
          regression_metrics.TyphoonTemporalRegressionMetrics(
              preds="batch.mean_future_pressures",
              targets="batch.pressure",
          )
      ),
  }

  # Training dataset.
  # custom_transform is used to apply model-specific preprocessing steps
  # eg. resizing for videoprism or normalization for imagenet_dinov2
  custom_transforms = None
  if "custom_transform" in cfg.aux:
    custom_transforms = cfg.aux["custom_transform"]

  cfg.aux["max_start_frame"] = 8

  # If debug is True, load additional fields in datasets for debugging purposes.
  cfg.aux["debug"] = False

  @konfig.ref_fn
  def get_train_ds(aux):
    return _make_ds(
        subset="train",
        shuffle=True,
        # Sample clips randomly from the input video for training.
        sampling="random_clip",
        max_start_frame=aux.max_start_frame,
        custom_transforms=custom_transforms,
        # We override train_ds defaults here following paper task definition and
        # baseline. Note that inputs will be further resized to match the
        # spatiotemporal resolution seen by the model during pretraining,
        # w. custom_transforms (but starting from the below settings).
        # - Following the task definition, we use 12 input frames.
        num_frames=typhoon.NUM_INPUT_FRAMES,
        # - We also follow the paper baseline setting for the spatial resolution
        im_size=(typhoon.IMAGE_RESOLUTION, typhoon.IMAGE_RESOLUTION),
        batch_size=aux.train_ds.batch_size,
        pred_len=pred_len,
        debug=cfg.ref.aux.debug,
    )

  cfg.train_ds = get_train_ds(cfg.ref.aux)

  # Training visualizations.
  show_images = lambda key: images.ShowSubsampledImages(
      images=key,
      num_images=3,  # Display 3 samples
      # Group consecutive frames horizontally.
      rearrange="... T H W c -> ... H (T W) c",
      # Subsample in the time dimension.
      subsample_dim=-4,
      subsample_step=4,
  )
  cfg.train_summaries = {
      # Display image model inputs.
      "model_input": show_images("batch.model_input"),
  }

  # Evaluations.
  # Set to True below to run the final eval on the test set.
  cfg.aux["run_eval_on_test"] = False

  @konfig.ref_fn
  def get_evals(aux) -> dict[str, kd.evals.Evaluator]:
    """Returns a dictionary of evaluation configurations."""
    eval_ds_cfg = dict(
        # Like for training, we override eval_ds_cfg defaults here following
        # paper task definition and baseline.
        num_frames=typhoon.NUM_INPUT_FRAMES,
        im_size=(typhoon.IMAGE_RESOLUTION, typhoon.IMAGE_RESOLUTION),
        batch_size=1,  # to avoid dropping samples.
    )

    evals = {
        "val": kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(aux.eval_interval_steps),
            num_batches=None,
            ds=_make_ds(
                subset="val",
                **eval_ds_cfg,
                custom_transforms=custom_transforms,
                num_epochs=1,
                pred_len=pred_len,
                debug=aux.debug,
            ),
        ),
    }
    if aux.run_eval_on_test:
      evals["test"] = kd.evals.Evaluator(
          run=kd.evals.Once(step=cfg.ref.num_train_steps),
          num_batches=None,
          ds=_make_ds(
              subset="test",
              **eval_ds_cfg,
              custom_transforms=custom_transforms,
              num_epochs=1,
              pred_len=pred_len,
              debug=aux.debug,
          ),
      )
    return evals

  cfg.evals = get_evals(cfg.ref.aux)

  return cfg
