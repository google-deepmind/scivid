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

"""Config for the models."""

from etils import epy
from kauldron import konfig


# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
with epy.lazy_imports():
  from scivid.configs.models import mock_model
  from scivid.configs.models import hf_videomae

with konfig.imports():
  from kauldron import kd
# pylint: enable=g-import-not-at-top


def update_config(cfg: kd.train.Trainer, name: str) -> kd.train.Trainer:
  """Returns updated config containing model config."""
  module = globals().get(name)
  if module is None:
    raise ValueError(f'Unknown model: {name}')
  return module.get_config(cfg)
