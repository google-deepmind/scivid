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

"""Base launch config for Scivid evals.

This launch config is parametrized by the model and eval name.
The parameters are passed in the command line as a colon-separated string e.g.
--cfg=launch_config:model_name:eval_name
"""

from absl import logging

from kauldron import konfig

from scivid.configs import eval_config
from scivid.configs import model_config

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  import optax
  from scivid.optim import utils as optim_utils
# pylint: enable=g-import-not-at-top

DEFAULT_BATCH_SIZE = 32  # DO NOT CHANGE - important for comparison
TOTAL_TRAIN_DATA_POINTS = 1_280_000  # DO NOT CHANGE - important for comparison
# For efficient online evaluation (alongside training), we limit the number
# of eval points. For certain benchmarks, this is a small subset of the
# validation set, so we call this evaluation setting "minival".
TOTAL_EVAL_DATA_POINTS = 4_096  # DO NOT CHANGE - important for comparison
# Number of steps between each run of the online evaluation ("minival" or just
# "val" if validation is small enough).
EVAL_INTERVAL_STEPS = 1000  # DO NOT CHANGE - important for comparison


def get_config(args: str | None = None) -> kd.train.Trainer:
  """Hyperparameter settings that are shared across all evals."""
  if args is None:
    logging.info('No args provided, using default model and eval.')
    args = 'mock_model:flyvsfly_classification'
  model_name, eval_name = args.split(':')

  cfg = kd.train.Trainer()
  cfg.seed = 0

  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=1_000,
      max_to_keep=10,
  )

  cfg.aux = {
      'lr': 3e-4,
      'train_ds': {
          'batch_size': DEFAULT_BATCH_SIZE,
          'im_size': (224, 224),
          'num_frames': 16,
      },
      # How deep to load readout from, in the backbone
      # (for models supporting it)
      'readout_depth_fraction': (0.95,),
      'backbone_lr_multiplier': 0.0,
      'weight_decay': 1e-4,
      'grad_clip': None,
  }
  cfg.aux.update({'finetune': cfg.ref.aux.backbone_lr_multiplier > 0.0})

  # Model should see same amount of training and evaluation data,
  # regardless of batch size
  cfg.num_train_steps = (
      TOTAL_TRAIN_DATA_POINTS // cfg.ref.aux.train_ds.batch_size
  )
  cfg.aux.num_eval_steps = (
      TOTAL_EVAL_DATA_POINTS // cfg.ref.aux.train_ds.batch_size
  )
  # For any efficient evaluation that we want to run alongside training:
  cfg.aux.eval_interval_steps = EVAL_INTERVAL_STEPS * (
      DEFAULT_BATCH_SIZE // cfg.ref.aux.train_ds.batch_size
  )
  # For any time-consuming evaluation that we want to run alongside training,
  # but less frequently:
  cfg.aux.long_eval_interval_steps = cfg.ref.aux.eval_interval_steps * 10

  @konfig.ref_fn
  def get_warmup_steps(num_train_steps):
    """Ensure warmup steps does not exceed num_train_steps to avoid ValueError."""
    return min(1000, num_train_steps - 1)

  cfg.schedules = {
      'learning_rate': optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=cfg.ref.aux.lr,
          warmup_steps=get_warmup_steps(cfg.ref.num_train_steps),
          decay_steps=cfg.ref.num_train_steps,
          end_value=1e-7,
      )
  }
  frozen = optim_utils.zero_grads()

  @konfig.ref_fn
  def get_optimizer(aux):
    """Returns the optimizer for the model.

    Add gradient clipping if grad_clip is not None and freeze the backbone if
    finetune is False.

    Args:
      aux: Auxiliary parameters.
    """
    optimizer = optax.adamw(
        learning_rate=cfg.ref.schedules['learning_rate'],
        weight_decay=cfg.ref.aux['weight_decay'],
    )
    # Add gradient clipping if specified.
    if aux['grad_clip'] is not None:
      optimizer = optax.chain(
          optimizer, optax.clip_by_global_norm(cfg.ref.aux['grad_clip'])
      )
    optimizer = optax.chain(
        optimizer,
        optim_utils.scale_backbone(cfg.ref.aux.backbone_lr_multiplier),
    )
    if aux['finetune']:
      return optimizer
    return optax.multi_transform(
        {'trained': optimizer, 'frozen': frozen},
        optim_utils.make_ignore_base_model_filter_fn(),
    )

  cfg.optimizer = get_optimizer(cfg.ref.aux)

  cfg = model_config.update_config(cfg, model_name)
  cfg = eval_config.update_config(cfg, eval_name)

  return cfg
