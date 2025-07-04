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

"""Evaluation config for STIR endoscopic point tracking.

Evaluation involves finetuning on Kubric and then running on STIR at test.

Notations:
  - B: Batch size
  - T: Number of frames
  - Q: Number of query points / tracks
  - C: Number of channels in feature vectors
  - N: Number of feature vectors

Expected shapes:
  - Dataset sample
    num_tracks = 34 (the maximum number of points that are tracked for
    each sample)
    - query_coords: [B, Q, 2]
        This is a list of query points in the first frame.
    - target_coords: [B, (T,) Q, 2]
        This is a list (per-frame) of ground truth target points. In practice,
      for STIR, all target points are only in the last frame so we omit the T
      dimension.
    - target_vis: [B, T, Q, 1]
        This is a mask to indicate which target points are visible in any frame.
      Only available in Kubric.
    - query_mask: [B, Q, 1]
        This is a mask to indicate which query points are valid. In Kubric, it
      is used to filter out tracks which are not visible at the query timestep.
      In STIR, query points are padded to a fixed length, the mask is set to 0
      for the padded points.
    - target_mask: [B, (T,) Q, 1]
        This is a mask to indicate which target points are valid. In Kubric, it
      is the intersection of the query mask and the target visibility. In STIR,
      the target point is only in the last frame so we omit the T dimension.
      Target points are padded to a fixed length, the mask is set to 0 for the
      padded points.
    - video: [B, T, H, W, 3]
  - Readout input
    - features: [B, T, N, C]
    - conditioning: [B, Q, 2]
  - Readout output
    - values: [B, T, Q, 2] (only last frame is used in STIR eval metric)
    - visible: [B, T, Q, 1] (used for training Kubric, ignored in STIR)
"""
from collections.abc import Sequence
import copy
from typing import Any


from kauldron import konfig


# pylint: disable=g-import-not-at-top
with konfig.imports():
  from flax import linen as nn
  from kauldron import kd
  from scivid.data.readers import movi
  from scivid.data.readers import stir_2d
  from scivid.data.transforms import point_tracking_transforms
  from scivid.data.transforms import tf_transforms
  from scivid.losses import tap as tap_losses
  from scivid.metrics import point_tracking
  from scivid.models import base_modules
  from scivid.models import pos_embeddings
  from scivid.models import readouts
  from scivid.models import readout_wrapper
  from scivid.summaries import images as image_summaries

# pylint: enable=g-import-not-at-top


def _make_kubric_ds(
    num_frames: int,
    query_timestep: int = 0,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    num_point_tracks: int = 64,
    im_size: tuple[int, int] = (128, 128),
    custom_transforms: list[Any] | None = None,
    video_len: int = 24,
    num_workers: int = 16,
):
  """Create MOVI point tracking dataset for training.

  Args:
    num_frames: The number of frames to sample from each video to form a clip.
    query_timestep: The index of the frame within the clip from which the query
      points are extracted.
    num_epochs: The number of times to iterate through the dataset. None means
      iterate indefinitely.
    batch_size: The number of video clips in each batch.
    num_point_tracks: The number of point tracks to extract per video clip.
      These points are sampled from the available ground truth tracks. This
      defines the number of query points and corresponding target points.
    im_size: The spatial resolution (height, width) to resize the video frames
      to.
    custom_transforms: A list of additional transformation functions to apply to
      the dataset elements.
    video_len: The original number of frames in the videos as stored in the
      dataset. This is used to load the correct data before any temporal
      sampling or padding to `num_frames`.
    num_workers: The number of worker processes to use for data loading.

  Returns:
    A dataset.
  """
  transforms = []

  if custom_transforms is None:
    custom_transforms = []

  transforms += [
      kd.data.Elements(
          keep=["video", "target_coords", "target_vis"],
      ),
      kd.data.ValueRange(key="video", in_vrange=(0, 255), vrange=(0, 1)),
  ]

  # If 'video' is longer than 'num_frames' then a random clip is extracted
  # from it. If it is shorter, it is padded with black frames.
  transforms += [
      tf_transforms.TemporalRandomWindow(
          key=["video", "target_coords", "target_vis"],
          length=num_frames,
      ),
  ]

  # Add point tracks
  transforms += [
      # Ground truth tracks are generated from random timesteps in the video.
      # Some of the tracks may be occluded at the query timestep. We mask these
      # tracks out as the queries need to be visible points.
      point_tracking_transforms.MaskOccludedTracks(
          tracks_key="target_coords",
          visible_key="target_vis",
          mask_key="query_mask",
          query_timestep=query_timestep,
      ),
      # Reshape target_vis to [..., T, Q, 1].
      kd.data.Rearrange(
          key=["target_vis"],
          pattern="... T Q -> ... T Q 1",
      ),
      # Combine target_vis and query_mask into a single mask for loss.
      point_tracking_transforms.CombineTrackMasksForLoss(
          target_vis_key="target_vis",
          query_mask_key="query_mask",
          target_mask_key="target_mask",
      ),
      # The target coords correspond to the full track. The query coords
      # correspond to the first frame points.
      kd.data.Gather(
          key={"target_coords": "query_coords"}, axis=0, indices=(0,)
      ),
      # We remove the temporal axis from the query coords.
      kd.data.Rearrange(key=["query_coords"], pattern="1 Q C -> Q C"),
  ]

  return movi.MOViPygrainReader(  # pytype: disable=wrong-keyword-args
      subset="train",
      shuffle=True,
      transforms=transforms + custom_transforms,
      batch_size=batch_size,
      num_epochs=num_epochs,
      im_size=im_size,
      # Load the full video, it will be subsampled in `TemporalRandomWindow`.
      num_frames=video_len,
      num_tracks=num_point_tracks,
      num_workers=num_workers,
  )


def _make_stir_ds(
    batch_size: int = 1,
    sampling: str = "linspace_clip",
    num_clips: int = 1,
    num_frames: int = -1,
    stride: int = 1,
    subset: str = "valid",
    im_size: tuple[int, int] = (640, 512),
    num_epochs: int | None = None,
    custom_transforms: Sequence[Any] | None = None,
) -> stir_2d.STIR2DPygrainReader:
  """Create STIR evaluation dataset."""

  transforms = [
      kd.data.Elements(
          keep=[
              "video",
              "query_coords",
              "target_coords",
              "query_mask",
              "target_mask",
              "video_idx",
              "duration",
          ],
      ),
      kd.data.ValueRange(key="video", vrange=(0, 1)),
  ]

  if custom_transforms is not None:
    for transform in custom_transforms:
      transforms.append(transform)

  return stir_2d.STIR2DPygrainReader(
      dataset_name="stir_2d_point_tracking",
      subset=subset,
      num_frames=num_frames,
      sampling=sampling,
      # As STIR only provides test sequences, we do not expect to use this
      # dataset for training.
      shuffle=False,
      batch_size=batch_size,
      im_size=im_size,
      stride=stride,
      seed=0,
      num_clips=num_clips,
      transforms=transforms,
      num_epochs=num_epochs,
  )


def get_points_loss(
    name="points",
    resolution=128,
    logits_visible_key="logits[..., :1]",
    logits_certainty_key="logits[..., 1:]",
):
  """Returns the loss for a point readout.

  Args:
    name: The name of the readout.
    resolution: The resolution of the points.
    logits_visible_key: The key of the logits for the visible prediction.
    logits_certainty_key: The key of the logits for the certainty.
  """

  readout_loss = {
      f"{name}_readout": tap_losses.Huber(
          pred_points=f"preds.readouts.{name}.values",
          target_points="batch.target_coords",
          delta=1.0 / resolution,
          mask="batch.target_mask",
          weight=100.0,
      ),
      f"{name}_readout_visible": kd.losses.SigmoidBinaryCrossEntropy(
          logits=f"preds.readouts.{name}.{logits_visible_key}",
          labels="batch.target_vis",
          mask="batch.query_mask[:, None]",
          weight=1e-1,
      ),
      f"{name}_readout_certainty": tap_losses.Certainty(
          logits=f"preds.readouts.{name}.{logits_certainty_key}",
          pred_points=f"preds.readouts.{name}.values",
          target_points="batch.target_coords",
          threshold=6.0 / resolution,
          mask="batch.target_mask",
          weight=1e-1,
      ),
  }
  return readout_loss


@konfig.ref_fn
def get_points_head(aux):
  """Returns the points head for STIR.

  The point tracking head consists of a cross-attention where queries are given
    by embeddings of the query positions, and keys and values are provided by
    the backbone features. The outputs are then mapped to 4D outputs holding the
    2D position prediction and the visibility and uncertainty estimates.

  Args:
    aux: A reference to the auxiliary config.
  """
  num_frames_per_query = (
      aux["train_ds"]["num_frames"] // aux["temporal_tile_size"]
  )
  return {
      "points": readouts.TrackingReadoutWrapper(
          attention_readout=readouts.AttentionReadout(
              # 2 for xy, 2 for visibility and uncertainty, times number of
              # frames divided by temporal tile size.
              num_classes=4 * num_frames_per_query,
              num_params=1024,
              num_heads=8,
          ),
          query_initializer=kd.nn.Sequential(
              layers=[
                  pos_embeddings.SampleFourierEmbedding(
                      num_fourier_bases=16,
                      update_type="replace",
                  ),
                  base_modules.MLP(
                      hidden_size=512,
                      output_size=512,
                  ),
              ]
          ),
          temporal_tile_size=aux["temporal_tile_size"],
          output_activation=nn.sigmoid,
          num_frames_per_query=num_frames_per_query,
          predict_visibility=True,
          use_certainty=True,
      ),
  }


def get_config(cfg: kd.train.Trainer) -> kd.train.Trainer:
  """The default hyperparameter configuration."""

  # Model.
  cfg.aux["temporal_tile_size"] = 8
  points_head = get_points_head(cfg.ref.aux)

  # temporal feature dimension assumed by the readout head
  # (which models must produce)
  cfg.aux["task_num_frames"] = cfg.ref.aux["train_ds"]["num_frames"]

  cfg.model = readout_wrapper.ReadoutWrapper(
      # Inner model
      model_inputs={cfg.aux["readout"]["model_inputs_key"]: "batch.video"},
      model=cfg.aux["readout"]["model"],
      # Readout heads
      readout_inputs={
          "points": {
              "inputs": cfg.aux["readout"]["readout_inputs"]["features"],
              "queries": "batch.query_coords",
          }
      },
      readout_heads=points_head,
      finetune=cfg.ref.aux["finetune"],
  )

  # Training losses.
  points_loss = get_points_loss(
      name="points",
      resolution=cfg.ref.aux["train_ds"]["im_size"][0],
      logits_visible_key="logits_visible",
      logits_certainty_key="logits_certainty",
  )
  cfg.train_losses = points_loss

  # Metrics.
  eval_metrics = {
      "attention.acc": point_tracking.STIR2DErrorPerTrackAssumingNoOcclusions(
          pred_coords="preds.readouts.points.values[..., -1, :, :]",
          query_mask="batch.query_mask",
          gt_coords="batch.target_coords",
          target_mask="batch.target_mask",
      ),
      "control.acc": point_tracking.STIR2DErrorPerTrackAssumingNoOcclusions(
          pred_coords="batch.query_coords",
          query_mask="batch.query_mask",
          gt_coords="batch.target_coords",
          target_mask="batch.target_mask",
      ),
  }
  cfg.train_metrics = {}

  # Training dataset.
  custom_transforms = []
  if "custom_transform" in cfg.aux:
    custom_transforms = cfg.aux["custom_transform"]

  # Kubric dataset loading
  @konfig.ref_fn
  def get_train_ds(aux):
    return _make_kubric_ds(
        custom_transforms=custom_transforms,
        **aux.train_ds,
    )

  cfg.train_ds = get_train_ds(cfg.ref.aux)

  # Training visualizations.
  cfg.train_summaries = {}
  cfg.train_summaries["video/gt"] = image_summaries.ShowTrackedPoints(
      videos="batch.video",
      tracks="batch.target_coords",
      visible="batch.target_vis",
      num_videos=2,
  )
  cfg.train_summaries["video/pred"] = image_summaries.ShowTrackedPoints(
      videos="batch.video",
      tracks="preds.readouts.points.values",
      # STIR doesn't come with occlusion information and assumes all points
      # remain visible throughout the video at eval time. We therefore display
      # all predicted final locations, instead of using the predicted
      # visibilities from the model.
      visible="batch.target_vis",
      num_videos=2,
  )

  # Evaluations.
  # Set to True below to run the final eval on the test set.
  cfg.aux["run_eval_on_test"] = False

  @konfig.ref_fn
  def get_evals(aux) -> dict[str, kd.evals.Evaluator]:
    """Returns a dictionary of evaluation configurations."""

    # Match train_ds config - except any fields we set below.
    eval_ds_cfg = copy.deepcopy(aux.train_ds)
    eval_ds_cfg.batch_size = 1  # to avoid dropping samples

    evals = {
        # Run eval on all the videos of the validation set, without duration
        # filtering.
        "val": kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(aux["eval_interval_steps"]),
            ds=_make_stir_ds(
                stride=1,
                num_epochs=1,
                subset="full",
                custom_transforms=custom_transforms,
                **eval_ds_cfg,
            ),
            metrics=eval_metrics,
            num_batches=None,
            losses={},
            summaries={},
        ),
    }
    if aux.run_eval_on_test:
      # Run eval on all 60 videos from the test set.
      evals["test"] = kd.evals.Evaluator(
          run=kd.evals.Once(step=cfg.ref.num_train_steps),
          ds=_make_stir_ds(
              stride=1,
              num_epochs=1,
              subset="test",
              custom_transforms=custom_transforms,
              **eval_ds_cfg,
          ),
          metrics=eval_metrics,
          num_batches=None,
          losses={},
          summaries={},
      )
    return evals

  cfg.evals = get_evals(cfg.ref.aux)

  return cfg
