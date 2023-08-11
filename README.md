## DIDA: Dual-level Interaction for Domain Adaptive Semantic Segmentation

----------------------------------------------------------------------------------------------------
This repository contains code and models for "DIDA: Dual-level Interaction for Domain Adaptive Semantic Segmentation" (accepted to ICCV Workshop on Uncertainty Quantification for Computer Vision (UnCV), 2023). [arXiv](https://arxiv.org/abs/2307.07972)

Copyright (c) 2022-2023 WHU China, Dongyu Yao. All rights reserved. 

Licensed under the Apache License, Version 2.0

----------------------------------------------------------------------------------------------------

For Chinese version of Instructions,  please refer to  [DIDA中文文档](README_zh.md)

## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/dida
source ~/venv/dida/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0  # requires the other packages to be installed first
```

Further, please download the MiT weights and a pretrained DIDA using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

All experiments were executed on a single NVIDIA RTX 3090.

## Inference Demo

Already as this point, the provided DIDA model (downloaded by
`tools/download_checkpoints.sh`) can be applied to a demo image:

```shell
sh demo.sh work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa.json work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/latest.pth
```

When judging the predictions, please keep in mind that DIDA had no access
to real-world labels during the training.

## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

The final folder structure should look like this:

```none
DIDA
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia (optional)
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

**Data Preprocessing:** Following DAFormer, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Training

A training job can be launched using:

```shell
sh run_config.sh [gpuid]
```

For the two experiments conducted in our paper, we use a system to automatically generate
and train the configs:
gta->cs: expid == 7
synthia->cs: expid == 8

```shell
sh run_exp.sh [expid] [gpuid]
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

## Testing & Predictions

### Manual Download

Please, download the [MiT weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) pretrained on ImageNet-1K provided by the official [SegFormer repository](https://github.com/NVlabs/SegFormer) and put them in a folder `pretrained/` within this project. For most of the experiments, only mit_b5.pth is necessary.

Please, download the checkpoint of DIDA on GTA->Cityscapes from [here](https://drive.google.com/file/d/1mw8mTui-I-mvs2vo0UN_xs_fQh7AMpHm/view?usp=sharing) and extract it to `work_dirs/`

The provided DIDA checkpoint trained on GTA→Cityscapes can be tested on the Cityscapes validation set using:

```shell
sh test.sh work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa
```

The predictions are saved for inspection to `work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/preds` and the mIoU of the model is printed at `pre_log.out`

The provided checkpoint should achieve 71.03 mIoU. Refer to the end of  `work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/20221213_101429.log`

for more information such as the class-wise IoU.

Similarly, also other models can be tested after the training has finished:

```shell
sh test.sh path/to/checkpoint_directory
```

When evaluating a model trained on Synthia→Cityscapes, please note that the
evaluation script calculates the mIoU for all 19 Cityscapes classes. However,
Synthia only contains labels for **16** of these classes. Therefore, it is a common
practice in UDA to report the mIoU for Synthia→Cityscapes only on these 16
classes. As the Iou for the 3 missing classes is 0, you can do the conversion
**mIoU16 = mIoU19 * 19 / 16**.

The predictions can be submitted to the public evaluation server of the
respective dataset to obtain the test score.

## Checkpoints

Below, we provide checkpoints of DIDA for different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the **median validation performance** here.

* [DIDA for GTA→Cityscapes](https://drive.google.com/file/d/1mw8mTui-I-mvs2vo0UN_xs_fQh7AMpHm/view?usp=sharing)
* [DIDA for Synthia→Cityscapes](https://drive.google.com/file/d/1fpAsxbhlIxzPhiIfm0r8rAoTjK4PZb_V/view?usp=share_link)

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU for 19 classes. For Synthia→Cityscapes, it is
  necessary to convert the mIoU to the 16 valid classes. Please, read the
  section above for converting the mIoU.

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0), the framework is partly based on [DAFormer](https://github.com/lhoyer/DAFormer).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html), the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html). and DAFormer readme file.

The most relevant files for DIDA are:

* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of UDA self-training, introduction of instance_loss, sampling strategies and dual-level interaction
* [embedding_cache](embedding_cache):
  Our pre-generated *instance bank*, with class-balanced distribution of instance features.
* [create_buffer.py](create_buffer.py):
  generate vacant placeholders and initiallize as unit random vectors

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
* [DAFormer](https://github.com/lhoyer/DAFormer)

## License

This project is released under the [Apache License 2.0](LICENSE), while some specific features in this repository are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for
commercial matters.

If you find our work helpful, please kindly cite us via the following bibtex:

@inproceedings{yao2023dual,
  title={Dual-level Interaction for Domain Adaptive Semantic Segmentation},
  author={Yao, Dongyu and Li, Boheng},
  booktitle={International Conference on Computer Vision Workshop on Uncertainty Quantification for Computer Vision},
  year={2023}
}

