We provide all the datasets in a unified format at the [SciVid GCS bucket](https://storage.googleapis.com/scivid),
and document below the transformations we made to the original datasets.

## Transformations applied to the datasets

__For Fly-vs-Fly__, following previous work [1, 2]:

- We filtered out all frames without valid trajectories.
We note that while video models can be ran on all frames, we do this to be
comparable to baselines evaluated in [1] and [2]. This uses the original
processing code provided by the Fly vs. Fly paper, which consists of the
following steps: for each frame in each video, remove trajectories (and
associated frames/labels) that contain NaN values. Only valid frames are used
for train/val/test.
- We only kept behaviors that had >1000 frames.
- We split the dataset at the video-level into clips and associated to each clip the label corresponding to the middle frame.
- We use the same train/val/test split as Task Programming [1],
which was split at the video-level and ensuring that each behavior exists
across all splits. The behaviors studied in [1,2] are lunge, wing threat,
tussle, wing extension, circle, copulation, and other (no behavior of interest).

__For CalMS21:__

- We only evaluate on task 1 since it is the only one with videos (following [2]).
- The original CalMS21 dataset only provided a train/test set;
to create a val split, we further split the CalMS21 train split into train and
val. The val split consists of 8 videos from the original train split, with all
behaviors present. These videos are the last few videos in the train set by ID:
063, 064, 065, 066, 067, 068, 069, 070.
- The original version of the dataset is extracted with a stride of 1 in the
time dimension, resulting in redundancies in input frames, which are
shared across successive clips. To reduce the total memory footprint of the
dataset, we subsample the train sets by 16 in the time axis.
- To further reduce the memory footprint, we downsampled the video from 1024x570 to 512x285.

__For Digital Typhoon:__

- We split the original dataset into fixed train / val / test subsets by applying random splitting at the sequence level. The resulting sets contain 696, 174 and 219 sequences respectively.
- Following [authors’ code](https://github.com/kitamoto-lab/benchmarks/blob/1bdbefd7c570cb1bdbdf9e09f9b63f7c22bbdb27/forecasting/Dataloader/SequenceDatamodule.py#L108) for dataset processing, we kept satellite images up to the year 2022 (included), with grade < 7, and longitudes within [100, 180].
- We downsampled the videos from 512x512 to 256x256 to reduce the memory footprint of the dataset.

__For Weatherbench 2:__
We took the publicly available ARCO ERA5 weather [dataset](gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1) as specified [here](gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1).

We then filtered this dataset to retain only the three key variables and levels (geopotential@500, temperature@850, and specific_humidity@700). Furthermore, we downsampled this dataset from the initial spatial resolution of 0.25°, down to a coarser 1° resolution: this corresponds to a spatial resolution of 181 × 360. We do this by slicing the dataset spatially by taking every 4th pixel from the original 0.25° resolution.
Finally, we reduce the dataset temporally to only include the dates that span
the range included in our train, validation and test splits.

We split the data following [3]: the train, validation and test splits consist of all trajectories for the years 1979-2017, 2018 and 2020, respectively. We consider temporal trajectories, where frames are taken with a 12 hour interval, and consisting of 16 frames as input to the model and 16 future frames required for prediction.

__For STIR:__

- We filter all videos for which there are no query points,
following authors’ [code](https://github.com/athaddius/STIRMetrics/blob/0b9be8b935dd6d548cae69ff46a85ce9f8a1d71f/src/datatest/write2dgtjson.py#L84).
- We normalize all query and target point coordinates by image dimensions.
- We use the train and validation splits [from the 2024 challenge](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared) and the later-released [test set](https://zenodo.org/records/14803158).
- We downsampled the videos to 512x512 to reduce the memory footprint of the dataset.

__For MOVI:__

- We pre-computed 2048 point tracks per video with coordinates normalized by image dimensions.
- We downsampled the videos to 256x256 to reduce the memory footprint of the dataset.

[1] Jennifer J. Sun, Ann Kennedy, Eric Zhan, David J. Anderson, Yisong Yue, and
Pietro Perona. Task programming: Learning data efficient behavior
representations. In CVPR, 2021.6163617

[2] Zhao, Long, et al. "Videoprism: A foundational visual encoder for video
understanding." arXiv preprint arXiv:2402.13217 (2024).

[3] Rasp, Stephan, et al. "Weatherbench 2: A benchmark for the next generation
of data‐driven global weather models." Journal of Advances in Modeling Earth
Systems 16.6 (2024): e2023MS004019.

## Citations
If you use any of these datasets, please remember to cite the original papers
that introduced them:

```
@inproceedings{eyjolfsdottir2014flyvsfly,
  title={Detecting social actions of fruit flies},
  author={Eyjolfsdottir, Eyrun and Branson, Steve and Burgos-Artizzu, Xavier P and Hoopfer, Eric D and Schor, Jonathan and Anderson, David J and Perona, Pietro},
  booktitle={ECCV},
  year={2014},
}
```

```
@inproceedings{sun2021calms21,
  title={The multi-agent behavior dataset: Mouse dyadic social interactions},
  author={Sun, Jennifer J and Karigo, Tomomi and Chakraborty, Dipam and Mohanty, Sharada P and Wild, Benjamin and Sun, Quan and Chen, Chen and Anderson, David J and Perona, Pietro and Yue, Yisong and others},
  booktitle={NeurIPS},
  year={2021},
}
```

```
@article{schmidt2024stir,
   title={Surgical Tattoos in Infrared: A Dataset for Quantifying Tissue Tracking and Mapping},
   journal={IEEE Transactions on Medical Imaging},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Schmidt, Adam and Mohareri, Omid and DiMaio, Simon P. and Salcudean, Septimiu E.},
   year={2024},
}
```

```
@article{rasp2024wb2,
  title={Weatherbench 2: A benchmark for the next generation of data-driven global weather models},
  author={Rasp, Stephan and Hoyer, Stephan and Merose, Alexander and Langmore, Ian and Battaglia, Peter and Russell, Tyler and Sanchez-Gonzalez, Alvaro and Yang, Vivian and Carver, Rob and Agrawal, Shreya and others},
  journal={Journal of Advances in Modeling Earth Systems},
  year={2024},
}
```

```
@inproceedings{kitamoto2023typhoon,
 author = {Kitamoto, Asanobu and Hwang, Jared and Vuillod, Bastien and Gautier, Lucas and Tian, Yingtao and Clanuwat, Tarin},
 booktitle = {NeurIPS},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 title = {Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling of Tropical Cyclones},
 year = {2023}
}
```

Note that to evaluate on STIR, we train on MOVI-e dataset v1.0.0 which was
generated using Kubric simulator:

```
@inproceedings{greff2022kubric,
  title={Kubric: A scalable dataset generator},
  author={Greff, Klaus and Belletti, Francois and Beyer, Lucas and Doersch, Carl and Du, Yilun and Duckworth, Daniel and Fleet, David J and Gnanapragasam, Dan and Golemo, Florian and Herrmann, Charles and others},
  booktitle={CVPR},
  year={2022}
}
```
