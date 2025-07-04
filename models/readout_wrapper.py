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

"""ReadoutWrapper module for adding readout heads to a model."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from flax import linen as nn
from jax import lax
from kauldron import kontext
from kauldron.typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True)
class ReadoutWrapper(nn.Module, kw_only=True):  # pytype: disable=invalid-function-definition
  """Wrapper for adding readout heads to a model.

  This model adds arbitrary readout heads to a given model. Readout heads are
  lightweight learned modules which map from pretrained features from an
  arbitrary model to task outputs.

  Check out scivid/colabs/demo.ipynb for an example building a full trainer
  config with ReadoutWrapper.


  Example usage:

  ```
  from kauldron import konfig
  with konfig.imports():
    from kauldron import kd
    from scivid.models.backbones import pass_through  # Replace with your model
    from scivid.models import readouts  # Add new readout heads here
    from scivid.models import readout_wrapper

  cfg = kd.train.Trainer()

  # Initialize simple pass-through model, replace with your own model.
  model = pass_through.PassThroughModel(patch_size=(1, 14, 14))

  # Wrap the model with ReadoutWrapper
  cfg.model = kd.contrib.nn.ReadoutWrapper(
      model=model,
      model_inputs={"image": "batch.video"},
      readout_inputs={"classification": {"inputs": "preds.features"},},
      readout_heads={"classification": readouts.LinearReadout(num_classes=10)},
      optimizer=...,  # Add your optimizer config here
      train_ds=...,  # Add your training dataset config here
      workdir="/path/to/workdir",
  )

  # Resolve the trainer config.
  konfig.resolve(cfg)
  ```


  Attributes:
    model: The model onto which the readout head should be attached.
    model_inputs: Optional dictionary of model input key paths. Used to fill the
      keyword arguments for the `model.__call__` function from the context. For
      example, if the model expects two inputs `images` and `masks` then
      `model_inputs={"images": "batch.images", "masks": "batch.masks"}` would
      ensure that `batch["images"]` and `batch["masks"]` are fed as inputs. If
      set to None then the model inputs are filled using the `kontext.Key`
      annotations of the model.
    readout_heads: Dict of arbitrary nn.Modules to be used as trainable readout
      heads.
    readout_inputs: Dict of optional dictionaries of input key paths for the
      readout head. Similar to `model_inputs`, but in addition to "batch" also
      has access to the model predictions under "preds.model" and the model
      intermediates in "interms.model". If set to None then the readout inputs
      are filled using the `kontext.Key` annotations of the readout head.
    finetune: Whether to finetune the model. Defaults to False in which case
      there is a stop_gradient between the model and the readout head.
  """

  model: nn.Module
  model_inputs: Optional[dict[str, str | None]] = None

  readout_heads: dict[str, nn.Module]
  readout_inputs: dict[str, dict[str, str | None] | None]

  finetune: bool = False

  @typechecked
  @nn.compact
  def __call__(self, **kwargs) -> Any:
    model_inputs = self._get_model_inputs(kwargs)
    model_preds = self.model(**model_inputs)

    if not isinstance(model_preds, dict):
      raise ValueError(
          'The provided model must return a dict of predictions to be'
          ' compatible with ReadoutWrapper.'
      )

    if 'readouts' in model_preds:
      raise ValueError(
          'Key `readouts` found in model predictions. This key is reserved for'
          ' readout heads when using ReadoutWrapper.'
      )

    readouts = {}
    for head in self.readout_heads:
      readout_inputs = self._get_readout_inputs(kwargs, model_preds, head)
      if not self.finetune:
        readout_inputs = lax.stop_gradient(readout_inputs)
      readout = self.readout_heads[head](**readout_inputs)
      readouts[head] = readout

    return model_preds | {'readouts': readouts}

  @staticmethod
  def _is_external_keypath(keypath: str | None) -> bool:
    if keypath is None:
      return False  # Treat optional keypaths as internal
    return kontext.Path.from_str(keypath)[0] not in ['interms', 'preds']

  def __kontext_keys__(self) -> dict[str, str | None]:
    # Forward all the model keys.
    model_keypaths = {f'model_{k}': v for k, v in self._model_keypaths.items()}

    # As well as the readout keys that refer to external context.
    readout_batch_keypaths = {}
    for head in self.readout_heads:
      readout_batch_keypaths.update({
          f'readout_{head}_{k}': v
          for k, v in self._get_readout_keypaths(head).items()
          if self._is_external_keypath(v)
      })
    return model_keypaths | readout_batch_keypaths

  @property
  def _model_keypaths(self) -> dict[str, str | None]:
    if self.model_inputs is None:
      return kontext.get_keypaths(self.model)
    else:
      return self.model_inputs

  def _get_readout_keypaths(self, head: str) -> dict[str, str | None]:
    if self.readout_inputs[head] is None:
      return kontext.get_keypaths(self.readout_heads[head])
    else:
      return self.readout_inputs[head]

  def _get_model_inputs(self, kwargs):
    # separate out the model kwargs
    return {
        k.removeprefix('model_'): v
        for k, v in kwargs.items()
        if k.startswith('model_')
    }

  def _get_readout_inputs(self, kwargs, preds, head):
    # separate out the readout batch keys from kwargs
    readout_batch_inputs = {
        k.removeprefix(f'readout_{head}_'): v
        for k, v in kwargs.items()
        if k.startswith(f'readout_{head}_')
    }
    # get other (non-batch) keys
    readout_other_keypaths = {
        k: v
        for k, v in self._get_readout_keypaths(head).items()
        if not self._is_external_keypath(v)
    }
    # resolve (non-batch) readout keys
    interms = self.scope.root.variables().get('intermediates', None)
    ctx = {'preds': preds, 'interms': interms}

    readout_other_inputs = kontext.resolve_from_keypaths(
        ctx, readout_other_keypaths
    )
    return readout_batch_inputs | readout_other_inputs
