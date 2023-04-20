# ---------------------------------------------------------------
# Copyright (c) 2022-2023 WHU China, Dongyu Yao. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

# Instructions for Manual Download:
#
# Please, download the [MiT weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)
# pretrained on ImageNet-1K provided by the official
# [SegFormer repository](https://github.com/NVlabs/SegFormer) and put them in a
# folder `pretrained/` within this project. For most of the experiments, only
# mit_b5.pth is necessary.
#
# Please, download the checkpoint of DIDA on GTA->Cityscapes from
# [here](https://drive.google.com/file/d/1mw8mTui-I-mvs2vo0UN_xs_fQh7AMpHm/view?usp=share_link).
# and extract it to `work_dirs/`

# Automatic Downloads:
set -e  # exit when any command fails
mkdir -p pretrained/
cd pretrained/
gdown --id 1d7I50jVjtCddnhpf-lqj8-f13UyCzoW1  # MiT-B5 weights
cd ../

mkdir -p work_dirs/
cd work_dirs/
gdown --id 1mw8mTui-I-mvs2vo0UN_xs_fQh7AMpHm  # DIDA on GTA->Cityscapes
unzip DIDA_gta2cs.zip
rm DIDA_gta2cs.zip
cd ../
