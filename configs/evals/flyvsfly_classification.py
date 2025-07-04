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

"""FlyVsFly multi-label classification of fly behavior."""

import copy
import functools

from kauldron import konfig


# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  from scivid.data.transforms import transforms as data_transforms
  from scivid.data.transforms import tf_transforms
  from scivid.data.readers import flyvsfly
  from scivid.models import readouts
  from scivid.models import readout_wrapper
  from scivid.metrics import mean_average_precision as map_metric
  from scivid.metrics import stats
# pylint: enable=g-import-not-at-top


def _make_ds(
    subset: str = "train",
    shuffle: bool = False,
    batch_size: int = 32,
    num_frames: int = 16,
    im_size: tuple[int, int] = (144, 144),
    custom_transforms=None,
    num_epochs: int | None = None,
):
  """Creates a dataset for training or evaluation.

  Args:
    subset: Which subset of the dataset to use, one of "train" / "val" / "test".
    shuffle: Whether to shuffle the dataset.
    batch_size: Number of samples in a batch.
    num_frames: Number of input frames to extract (may be further resampled by
      the model's custom transforms to match the pretraining clip duration).
    im_size: Spatial resolution of the clip (may be further resized by the
      model's custom transforms to match the pretraining resolution).
    custom_transforms: A list of custom data transforms specific to the model
      being evaluated.
    num_epochs: The number of epochs to repeat the dataset. If None, repeats the
      dataset indefinitely.

  Returns:
    A dataset for training or evaluation.
  """

  transforms = [
      kd.data.Elements(keep=[], rename={"label": "labels", "image": "video"}),
      kd.data.ValueRange(key="video", vrange=(0, 1)),
      data_transforms.OneHot(key="labels", num_classes=flyvsfly.NUM_CLASSES),
      kd.data.Cast(key="labels", dtype="int32"),
  ]
  # Maybe add training-specific transforms, based on subset name.
  add_training_transforms = subset == "train"
  if add_training_transforms:
    transforms += [
        # Spatial jittering: slightly increase the resolution of the video
        # randomly and crop back to the original resolution.
        tf_transforms.RandomResize(
            key="video",
            prob=1.0,
            min_scale_factor=1.0,
            max_scale_factor=1.4,
        ),
        tf_transforms.RandomCrop(
            key="video", shape=(None, int(im_size[0]), int(im_size[1]), None)
        ),
    ]

  if custom_transforms:
    transforms += custom_transforms

  return flyvsfly.FlyvsFlyPygrainReader(
      dataset_name="flyvsfly",
      subset=subset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_frames=num_frames,
      im_size=im_size,
      seed=kd.random.PRNGKey(42),
      transforms=transforms,
      num_epochs=num_epochs,
  )


def get_config(cfg: kd.train.Trainer) -> kd.train.Trainer:
  """The default hyperparameter configuration."""

  # Model.
  model_inputs = {
      cfg.aux["readout"]["model_inputs_key"]: "batch.video",
  }
  cfg.aux["readout_num_params"] = 128
  cfg.aux["readout_num_heads"] = 16

  readout_model = readout_wrapper.ReadoutWrapper(
      # Inner model
      model_inputs=model_inputs,
      model=cfg.aux["readout"]["model"],
      # Readout heads
      readout_inputs={
          "linear_classification": {
              "inputs": cfg.aux["readout"]["readout_inputs"]["features"]
          },
          "attn_classification": {
              "inputs": cfg.aux["readout"]["readout_inputs"]["features"]
          },
      },
      readout_heads={
          "linear_classification": readouts.LinearReadout(
              flyvsfly.NUM_CLASSES,
              reduce_axis=(-2, -3),
          ),
          "attn_classification": readouts.AttentionReadout(
              num_classes=flyvsfly.NUM_CLASSES,
              num_params=cfg.ref.aux["readout_num_params"],
              num_heads=cfg.ref.aux["readout_num_heads"],
              dropout_rate=0.5,
          ),
      },
      finetune=cfg.ref.aux["finetune"],
  )

  cfg.model = readout_model

  # Readout heads do not assume a particular number of frames as input.
  # (no need to resize output of the model over temporal axis)
  cfg.aux["task_num_frames"] = None

  # Training losses.
  cfg.train_losses = {
      "xent_linear_head": kd.losses.SigmoidBinaryCrossEntropy(
          logits="preds.readouts.linear_classification", labels="batch.labels"
      ),
      "xent_attn_head": kd.losses.SigmoidBinaryCrossEntropy(
          logits="preds.readouts.attn_classification", labels="batch.labels"
      ),
  }
  cfg.aux["weight_decay"] = 0.0

  # Metrics.
  map_metric_fn = functools.partial(
      map_metric.MeanAveragePrecision,
      class_names=flyvsfly.CLASSES,
      skip_background=True,  # background is always skipped for fly-vs-fly.
      background_class_index=-1,
      # Further measure performance when skipping rare classes (tussling and
      # wing_threat). This is particularly useful for online eval where we have
      # very few samples for these classes and performance can vary a lot
      # across random seeds.
      skip_classes=(1, 2),
  )
  cfg.train_metrics = {
      "linear.mAP": map_metric_fn(
          predictions="preds.readouts.linear_classification",
          labels="batch.labels",
      ),
      "attention.mAP": map_metric_fn(
          predictions="preds.readouts.attn_classification",
          labels="batch.labels",
      ),
      "control.mAP": map_metric_fn(
          predictions="batch.label_frequencies_train",
          labels="batch.labels",
      ),
      "count": stats.Count(
          labels="batch.labels",
          class_names=flyvsfly.CLASSES,
      ),
  }

  # Training dataset.
  # custom_transform is used to apply model-specific preprocessing steps
  # eg. resizing for videoprism or normalization for imagenet_dinov2
  custom_transforms = None
  if "custom_transform" in cfg.aux:
    custom_transforms = cfg.aux["custom_transform"]

  @konfig.ref_fn
  def get_train_ds(aux):
    return _make_ds(
        shuffle=True,
        custom_transforms=custom_transforms,
        **aux.train_ds,
    )

  cfg.train_ds = get_train_ds(cfg.ref.aux)

  # Training visualizations.
  show_images = lambda key: kd.summaries.ShowImages(
      images=key,
      num_images=3,  # Display 3 samples
      # Group consecutive frames horizontally.
      rearrange="... T H W c -> ... H (T W) c",
  )
  cfg.train_summaries = {"video": show_images("batch.video")}

  # Evaluations.
  # Set to True below to run the final eval on the test set.
  cfg.aux["run_eval_on_test"] = False

  @konfig.ref_fn
  def get_evals(aux) -> dict[str, kd.evals.Evaluator]:
    """Returns a dictionary of evaluation configurations."""
    # Match train_ds config - except any fields we set below.
    eval_ds_cfg = copy.deepcopy(aux.train_ds)
    evals = {
        # Lightweight online evaluation, on a subset of the validation set.
        "minival": kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(aux.eval_interval_steps),
            ds=_make_ds(
                custom_transforms=custom_transforms,
                subset="val",
                **eval_ds_cfg,
                num_epochs=1,  # to collect each sample exactly once
            ),
            num_batches=aux.num_eval_steps,
        ),
        # Less frequent (bc time-consuming) eval on the full validation set.
        "val": kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(aux.long_eval_interval_steps),
            ds=_make_ds(
                custom_transforms=custom_transforms,
                subset="val",
                **eval_ds_cfg,
                num_epochs=1,  # to collect each sample exactly once
            ),
        ),
    }
    if aux.run_eval_on_test:
      # Maybe run the evaluation at the end of training on the test set.
      # (dropping the last incomplete batch).
      evals["test"] = kd.evals.Evaluator(
          run=kd.evals.Once(step=cfg.ref.num_train_steps),
          ds=_make_ds(
              custom_transforms=custom_transforms,
              subset="test",
              **eval_ds_cfg,
              num_epochs=1,  # to collect each sample exactly once
          ),
      )
    return evals

  cfg.evals = get_evals(cfg.ref.aux)

  return cfg
