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

"""Config for the evals."""

import importlib
import types

from kauldron import konfig


# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
# pylint: enable=g-import-not-at-top

SCIVID_CONFIGS = (
    'calms21_classification',
    'flyvsfly_classification',
    'stir_2d_tracking',
    'typhoon_future_pred',
    'weatherbench_future_pred',
)


def import_eval_module(name: str) -> types.ModuleType:
  """Import the eval module with the given name."""
  if name in SCIVID_CONFIGS:
    module = importlib.import_module(f'scivid.configs.evals.{name}')
  else:
    raise ValueError(f'Unknown eval: {name}')
  return module


def update_config(cfg: kd.train.Trainer, name: str) -> kd.train.Trainer:
  """Returns updated config containing eval config."""
  return import_eval_module(name).get_config(cfg)


def get_eval_module_names() -> tuple[str, ...]:
  return SCIVID_CONFIGS
