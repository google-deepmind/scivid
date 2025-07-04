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

r"""WeatherBench 2 (WB2) PyGrain-based data reader."""

import dataclasses
import functools
import os
from typing import Any, Optional, SupportsIndex
import warnings

from grain import python as grain
import jax
from kauldron.data.py import base as pybase
from kauldron.random import random
import numpy as np
import tensorflow as tf
import xarray
import xarray_tensorstore

GRID_RESOLUTION = 1  # resolution of the grid in degrees.
# IMAGE_DIMENSIONS should be set accordingly with GRID_RESOLUTION;
# eg. 1 deg means using the grid (-90:91, 0:361), of dimensions 181x360.
IMAGE_HEIGHT = 181
IMAGE_WIDTH = 360
IMAGE_DIMENSIONS = (IMAGE_HEIGHT, IMAGE_WIDTH)

INPUT_STEPS = 16  # number of input frames
TARGET_STEPS = 16  # (maximum) number of target frames to evaluate model over

# Time step between consecutive frames (in hours)
# Note: all eval settings used by graphcast, gencast and WB2 ERA5 use 12 hours;
# graphcast uses a model step of 6h but evaluates every other prediction.
TIMESTEP_HOURS = 12

WEATHERBENCH_FILE = 'weatherbench/weatherbench2_3variable_dataset.zarr'
# If SCIVID_DATA_DIR is set, use it as the local data directory.
if os.getenv('SCIVID_DATA_DIR'):
  ZARR_PATH = os.path.join(
      os.getenv('SCIVID_DATA_DIR'), 'full', WEATHERBENCH_FILE
  )
else:
  ZARR_PATH = os.path.join('gs://scivid/full', WEATHERBENCH_FILE)
COMMON_DATA_KWARGS = dict(
    # dataset path
    zarr_path=ZARR_PATH,
    # Timestep at which the underlying zarr dataset is stored.
    zarr_timestep_hours=1,
)

# Leave out 1979-01-01 for historical reasons.
START_TRAIN_DATE = '1979-01-02'
EARLIEST_EVAL_YEAR = 2018
TRAIN_SPLIT = f'date_range_{START_TRAIN_DATE}_{EARLIEST_EVAL_YEAR-1}'
TRAIN_DATA_KWARGS = dict(
    split=TRAIN_SPLIT,
    # Offset applied to the start time of contiguous
    # sequences subsampled from each time slice section of the dataset.
    temporal_offset_hours=0,
    # Train on trajectories offset by 6 hours from each other.
    temporal_stride_hours=6,
    **COMMON_DATA_KWARGS,
)

# Mini validation setting.
# Note: this setting (somewhat informally) considers all (input, target)
# sequences starting and ending in 2018.
# As a result, the number of samples depends on the sequence length.
ONLINE_DEV_YEAR = 2018
ONLINE_DEV_SPLIT = f'date_range_{ONLINE_DEV_YEAR}_{ONLINE_DEV_YEAR}'

assert ONLINE_DEV_YEAR >= EARLIEST_EVAL_YEAR

ONLINE_DEV_DATA_KWARGS = dict(
    split=ONLINE_DEV_SPLIT,
    # Offset applied to the start time of contiguous
    # sequences subsampled from each time slice section of the dataset.
    temporal_offset_hours=6,
    # 78 examples for a year's worth of data ((365.2 - (16+16)*12/24) // 4.5)+1
    temporal_stride_hours=108,  # ie every 4.5 days
    **COMMON_DATA_KWARGS,
)

# Full validation setting (from GraphCast paper).
# Includes all *forecast init times* in 2018 (ie with input sequences
# potentially extending into 2017, and target sequences into 2019); as a result,
# the number of samples only depends on the eval time step.
# However we need to set the split definition consistently with
# {INPUT/TARGET}_STEPS, because of the way the underlying data readers work.
OFFLINE_DEV_YEAR = 2018
OFFLINE_DEV_SPLIT = 'date_range_2017-12-24_2019-01-08'
assert OFFLINE_DEV_YEAR >= EARLIEST_EVAL_YEAR

OFFLINE_DEV_DATA_KWARGS = dict(
    split=OFFLINE_DEV_SPLIT,
    # Offset applied to the start time of contiguous
    # sequences subsampled from each time slice section of the dataset.
    temporal_offset_hours=18,  # so that 1st forecast init time is 01-01-060000
    temporal_stride_hours=12,
    **COMMON_DATA_KWARGS,
)

# Test evaluation setting (from WB2 paper).
# Includes all *forecast init times* in 2020 (as above, number of samples only
# depends on the eval time step, and the split definition has been set
# consistently with {INPUT/TARGET}_STEPS.)
TEST_YEAR = 2020
TEST_SPLIT = 'date_range_2019-12-24_2021-01-08'

assert TEST_YEAR >= EARLIEST_EVAL_YEAR

TEST_DATA_KWARGS = dict(
    split=TEST_SPLIT,
    # Offset applied to the start time of contiguous sequences subsampled from
    # each time slice section of the dataset.
    temporal_offset_hours=12,  # so that 1st forecast init time is 01-01-000000
    temporal_stride_hours=12,
    **COMMON_DATA_KWARGS,
)

# Specific dimensions and levels we use to form 3-channel input and target
# sequences of frames - chosen based on Weatherbench 2 headline scores;
# see https://arxiv.org/abs/2308.15560: Table 3.
GEOPOTENTIAL_LEVEL = 500  # hPa
TEMPERATURE_LEVEL = 850  # hPa
SPECIFIC_HUMIDITY_LEVEL = 700  # hPa
FILTERED_VARIABLES_AND_LEVELS = {
    'multilevel_temporal': [  # DataArray name where the values are stored.
        dict(
            multilevel_temporal_variable='geopotential',
            level=GEOPOTENTIAL_LEVEL,
        ),
        dict(
            multilevel_temporal_variable='temperature',
            level=TEMPERATURE_LEVEL,
        ),
        dict(
            multilevel_temporal_variable='specific_humidity',
            level=SPECIFIC_HUMIDITY_LEVEL,
        ),
    ]
}

NUM_FILTERED = 3  # We keep three variables, geopotential_500, temperature_850,
# specific_humidity_700. This should be consistent with the length of the
# variables_list in FILTERED_VARIABLES_AND_LEVELS.

# Statistics of the filtered variables.
# We store them here for use by the normalization / unnormalization processors,
# in the order: geopotential_500, temperature_850, specific_humidity_700
# (consistent with the ordering of channels in the 'video' data array).
FILTERED_MEAN = (54088.706200122775, 274.38956414978514, 0.002414660515267107)
FILTERED_STD = (3358.6479264235377, 15.691516333086565, 0.0025397288441720684)
# Weather forecasting approaches use the following parameterization:
#   y = eulerian_persistence(x) + residual * residual_std + residual_mean
# where
#   * eulerian_persistence(x) is the last frame of input x
#   * residual is the predictor output eg. readout(model(normalize(x)))
#   * residual_{mean,std} are the statistics of (target - last_frame_of_input)
# We store below these statistics to use them in our unnormalization processors.
# Following weather's InputsAndResiduals which sets use_residual_locations to
# False by default, we don't use the actual residual means, so we set them to 0.
FILTERED_RESIDUAL_MEAN = (0.0, 0.0, 0.0)
FILTERED_RESIDUAL_STD = (
    404.18805465281133,
    2.457512144662011,
    0.0012086132599619071,
)


def xarray_dataset_from_zarr(
    zarr_path: str,
    split: str,
    # TensorStore is a way to load zarr files, which is typically faster.
    # However, *it might break* with future versions of Xarray. See
    # https://github.com/google/xarray-tensorstore
    # We set the default to False to be safe, although in the code itself we
    # call this function with use_xarray_tensorstore = True.
    use_xarray_tensorstore: bool = False,
) -> xarray.Dataset:
  """Returns an Xarray dataset from a zarr file."""
  if use_xarray_tensorstore:
    ds = xarray_tensorstore.open_zarr(zarr_path)
  else:
    ds = xarray.open_zarr(zarr_path)

  date_start, date_end = split.removeprefix('date_range_').split('_')
  ds = ds.sel(time=slice(date_start, date_end))

  return ds


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class PyGrainEra5Reader(pybase.DataSourceBase):
  """WeatherBench 2 ERA5 PyGrain reader.

  We follow where appropriate the PygrainVideoReader interface (eg. for
  argument names and behavior).

  Attributes:
    subset: Subset of the dataset to use - usually train, valid, or test.
    input_steps: Number of input frames to use.
    target_steps: Number of target frames to use.
    timestep_hours: Time step between consecutive frames (in hours).
    shuffle: Whether or not to shuffle the dataset.
    num_epochs: Number of epochs to repeat the dataset.
  """

  subset: str
  # Clip sampling arguments
  input_steps: int = 16
  target_steps: int = 16
  timestep_hours: int = 12
  # Data loading arguments
  shuffle: bool = True
  num_epochs: Optional[int] = None

  # Number of workers to use for data loading. Overriding the default value from
  # the parent class. We set this to 1 explicitly to avoid out of memory and
  # possible multiprocess issues.
  num_workers: int = 1

  def __post_init__(self):
    # We support multi-host setups, but we warn the user if they are using more
    # than one host.
    num_hosts = jax.process_count()
    if num_hosts > 1:
      warnings.warn(
          f'Got {num_hosts} hosts. Make sure that the sharding of the data is'
          ' done correctly across hosts (see `ds_for_current_process` in the'
          ' `PyGrainEra5Reader` class) to avoid data repetition.',
          UserWarning,
          stacklevel=2,  # show the line where the warning was triggered.
      )
    if self.num_workers > 1:
      warnings.warn(
          f'Got {self.num_workers} workers. Please note that increasing the'
          ' number of workers above 1 may lead to out of memory issues,'
          ' especially when using large batch sizes. Consider reducing'
          ' the `num_workers` parameter in the `PyGrainEra5Reader` class or'
          ' reducing the `batch_size` in the Weatherbench eval config.',
          UserWarning,
          stacklevel=2,  # show the line where the warning was triggered.
      )

    # Ensure that GRID_RESOLUTION and IMAGE_DIMENSIONS are consistent.
    min_lat, max_lat = -90, 90  # lat -90 != lat 90
    min_lon, max_lon = 0, 359  # lon 360 == lon 0
    step = GRID_RESOLUTION

    num_lat_coords = len(range(min_lat, max_lat + step, step))
    num_lon_coords = len(range(min_lon, max_lon + step, step))

    if (num_lat_coords, num_lon_coords) != IMAGE_DIMENSIONS:
      raise ValueError(
          'GRID_RESOLUTION and IMAGE_DIMENSIONS are inconsistent.'
          f' {num_lat_coords} x {num_lon_coords} != {IMAGE_DIMENSIONS}'
      )

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    input_duration_hours = self.input_steps * self.timestep_hours
    target_duration_hours = self.target_steps * self.timestep_hours

    if self.subset == 'train':
      data_kwargs = TRAIN_DATA_KWARGS
    elif self.subset == 'online_dev':
      data_kwargs = ONLINE_DEV_DATA_KWARGS
    elif self.subset == 'offline_dev':
      data_kwargs = OFFLINE_DEV_DATA_KWARGS
    elif self.subset == 'test':
      data_kwargs = TEST_DATA_KWARGS
    else:
      raise ValueError(f'Unknown subset: {self.subset}')

    ds = WBPygrainDataSource(
        zarr_path=data_kwargs['zarr_path'],
        split=data_kwargs['split'],
        sequence_length_hours=input_duration_hours + target_duration_hours,
        timestep_hours=self.timestep_hours,
        temporal_stride_hours=data_kwargs['temporal_stride_hours'],
        temporal_offset_hours=data_kwargs['temporal_offset_hours'],
        zarr_timestep_hours=data_kwargs['zarr_timestep_hours'],
        use_xarray_tensorstore=True,
    )

    return ds

  def process_example(self, example: dict[str, Any]) -> dict[str, Any]:
    """Processes a single example from the dataset.

    Args:
      example: A tuple containing the output of the WeatherBench reader.

    Returns:
      A dict of processed 'image' (input frames) and 'future' frames.
    """
    video_array = example['video'].data

    # Transpose arrays so that time is in the first dimension and channels are
    # in the last dimension.
    # (C, T, H, W) -> (T, H, W, C)
    inputs_and_targets = tf.transpose(video_array, (1, 2, 3, 0))

    # Select the correct time steps for the input and output variables.
    num_target_frames = self.target_steps
    num_input_frames = self.input_steps
    input_frames, target_frames = tf.split(
        inputs_and_targets,
        [num_input_frames, num_target_frames],
        axis=0,
    )  # each of shape T{',''} x H x W x 3

    output = {
        'image': input_frames,
        'future': target_frames,
    }
    return output

  def stack_variables(self, ds: xarray.Dataset) -> xarray.Dataset:
    """Orders and stacks predefined variables into a single DataArray.

    This method assumes that the input `ds` already contains DataArrays
    named according to the patterns like 'geopotential_500', 'temperature_850',
    etc., as defined in `FILTERED_VARIABLES_AND_LEVELS`.

    It stacks these variables into a new DataArray named 'video', along a
    new dimension called 'stacked_variables'. The order of stacking is
    determined by `FILTERED_VARIABLES_AND_LEVELS`.
    Args:
      ds: The input xarray.Dataset.

    Returns:
      The xarray.Dataset with a new 'video' DataArray.
    """
    # Determine the target variable names in the desired order.
    ordered_target_vars = []
    for _, var_spec_list in FILTERED_VARIABLES_AND_LEVELS.items():
      for spec in var_spec_list:
        # Construct the target variable name, e.g., "geopotential_500"
        var_name = f"{spec['multilevel_temporal_variable']}_{spec['level']}"
        ordered_target_vars.append(var_name)

    # The dataset `ds` is assumed to contain DataArrays with the required names.
    # We stack them using `ordered_target_vars` to ensure the variables
    # are stacked in the correct order. If `ds` only contains these variables,
    # this only enforces order. If `ds` contains other variables, this step
    # also ensures only the specified ones are chosen.
    ds_subset_for_stacking = ds[ordered_target_vars]

    # Convert the ordered DataArrays in `ds_subset_for_stacking` into a
    # single DataArray.
    stacked_da = ds_subset_for_stacking.to_array(dim='selected_variable')

    # Add the new DataArray to the dataset as 'video'.
    # This returns a new dataset. If 'video' already existed, it's replaced.
    # Other variables/coordinates in the original `ds` are preserved. This is
    # the key that `process_example` will look for.
    return ds.assign(video=stacked_da)

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    ds = grain.MapDataset.source(self.data_source)
    ds = ds.seed(rng.as_seed())

    # Shard the dataset across different hosts.
    ds = ds[jax.process_index() :: jax.process_count()]

    # Global shuffle
    if self.shuffle:
      ds = ds.shuffle(seed=rng.fold_in('shuffle').as_seed())

    ds = ds.map(self.stack_variables)

    # Map the processing function
    ds = ds.map(self.process_example)
    return ds


class WBPygrainDataSource(grain.RandomAccessDataSource[xarray.Dataset]):
  """WeatherBench2 pygrain-based video data source for Kauldron."""

  def __init__(
      self,
      zarr_path: str,
      split: str,
      sequence_length_hours: int,
      timestep_hours: int,
      temporal_stride_hours: int = 1,
      temporal_offset_hours: int = 0,
      zarr_timestep_hours: int = 1,
      use_xarray_tensorstore: bool = False,
  ):
    """Initializes the WBPygrainDataSource.

    This data source loads weather data from a Zarr archive, typically
    representing a time series of weather variables. It enables efficient (lazy)
    extraction of temporal windows from this data.

    Args:
      zarr_path: The path to the Zarr dataset.
      split: The date range to load (e.g., "date_range_2019-12-24_2021-01-08"),
        "test"). This is passed to the Zarr loader to slice the dataset.
      sequence_length_hours: The total temporal duration (in hours) that each
        output data window (sequence) should span.
      timestep_hours: The temporal resolution (in hours) of the data within each
        output window. For example, a value of 6 means each time point in an
        output window will be 6 hours apart.
      temporal_stride_hours: The step size (in hours) to move forward when
        generating subsequent distinct windows from the dataset. This determines
        the difference between the start of one window and the start of the
        next.
      temporal_offset_hours: An initial offset (in hours) from the beginning of
        the start date of the split, before the first window is extracted. This
        can be used to skip an initial segment of the data.
      zarr_timestep_hours: The native temporal resolution (in hours) of the
        underlying data stored in the Zarr archive. This is the time difference
        between consecutive data points as they exist in the input Zarr store.
      use_xarray_tensorstore: A flag indicating whether to use xarray's
        TensorStore integration for opening the Zarr dataset. Using TensorStore
        can be faster, but it might break with future versions of Xarray.
    """

    # Data source for the current split. Note that xarray.Dataset is a lazy
    # loading object, so this does not load the entire dataset into memory.
    self.dataset: xarray.Dataset = xarray_dataset_from_zarr(
        zarr_path=zarr_path,
        split=split,
        use_xarray_tensorstore=use_xarray_tensorstore,
    )
    # The length of each output window.
    self._output_window_length: int = sequence_length_hours // timestep_hours

    # The stride within a single output window from the dataset.
    self._output_window_stride: int = timestep_hours // zarr_timestep_hours

    # The stride between windows in the zarr dataset.
    self._stride_between_windows: int = (
        temporal_stride_hours // zarr_timestep_hours
    )

    # The offset of the first window in the current split.
    self._first_window_offset: int = (
        temporal_offset_hours // zarr_timestep_hours
    )

    # The total length of the dataset. This is used to determine the number of
    # distinct windows in the current split.
    self.total_length: int = self.dataset.sizes['time']

    # The number of distinct windows in the current split. This determines the
    # number of distinct *data points* (for training/evaluation) in the current
    # split.
    self.num_distinct_windows: int = self._length_to_num_distinct_windows()

  def _length_to_num_distinct_windows(self) -> int:
    effective_length = self.total_length - self._first_window_offset
    num_total_windows = np.maximum(
        effective_length
        - (self._output_window_length - 1) * self._output_window_stride,
        0,
    )
    # The following is equivalent to
    # ceil(num_total_windows/stride_between_windows)
    return (
        num_total_windows + self._stride_between_windows - 1
    ) // self._stride_between_windows

  def __len__(self) -> int:
    """Returns the number of distinct windows in the dataset."""
    return self.num_distinct_windows

  def _extract_window(self, start_time_index: int) -> xarray.Dataset:
    """Extracts a time window from the dataset."""
    slice_ = slice(
        start_time_index,
        start_time_index
        + self._output_window_length * self._output_window_stride,
        self._output_window_stride,
    )

    dataset_window = self.dataset.isel(time=slice_)
    return dataset_window

  def __getitem__(self, record_key: SupportsIndex) -> xarray.Dataset:
    """Retrieves a record (time window) for the given record key (window index)."""
    if isinstance(record_key, int):
      if 0 <= record_key < self.num_distinct_windows:
        start_time_index = (
            self._first_window_offset
            + record_key * self._stride_between_windows
        )
        return self._extract_window(start_time_index)
      else:
        raise IndexError(
            f'Index {record_key} out of range for'
            f' {self.__class__.__name__} with {len(self)} windows.'
        )
    else:
      raise TypeError(
          f'Indexing with type {type(record_key)} is not supported. '
          'Expected an integer index.'
      )
