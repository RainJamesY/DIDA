## DIDA——双层级交互的领域自适应语义分割算法
---------------------------------------------------------------
版权所有 （c） 2022-2023 中国 武汉大学 ，姚栋宇。保留所有权利。

根据 Apache 许可证 2.0 版获得许可

----------------------------------------------------------------------------------------------------



## 配置环境

对于这个项目，我们使用了python 3.8.5。我们建议您配置新的虚拟环境：

```shell
python -m venv ~/venv/dida
source ~/venv/dida/bin/activate
```

在该环境中，可以使用以下命令安装要求内容：

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0  # requires the other packages to be installed first
```

此外，请使用以下脚本下载 MiT 权重和预训练的 DIDA。如果自动下载出现问题，请按照以下操作脚本中的说明手动下载。

```shell
sh tools/download_checkpoints.sh
```

我们的所有实验在单块 NVIDIA RTX 3090 GPU上进行.

## 推理展示

现在，我们提供的DIDA模型 (通过
`tools/download_checkpoints.sh` 命令下载) 已可应用于演示图像：

```shell
sh demo.sh work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa.json work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/latest.pth
```

在评价模型的预测结果时, 请您注意：DIDA在训练过程中没有用到真实世界的数据。

## 配置数据集

**Cityscapes:** 
  请点击[这里](https://www.cityscapes-dataset.com/downloads/)下载 leftImg8bit_trainvaltest.zip 和 gt_trainvaltest.zip 并将它们解压到 `data/cityscapes`。


**GTA:** 请点击[这里](https://download.visinf.tu-darmstadt.de/data/from_games/)下载所有图片和标签包，并将它们解压到 `data/gta`

**Synthia (可选项):** 
请点击[这里](http://synthia-dataset.net/downloads/)下载SYNTHIA-RAND-CITYSCAPES数据集并将其解压到`data/synthia`

最后的文件夹结构应如下图所示:

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

**数据预处理:** 与DAFormer一致，请运行以下脚本将标签 ID 转换为训练 ID 并为 RCS 生成类索引:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```



## 训练

以下步骤可启动一次训练任务:

```shell
sh run_config.sh [gpuid]
```

对于我们论文中进行的两个实验，我们使用一个系统来自动生成并训练配置：
gta->cs: expid == 7
synthia->cs: expid == 8

```shell
sh run_exp.sh [expid] [gpuid]
```

有关可用实验及其分配的 ID 的详细信息，请查看[experiments.py](experiments.py)，生成的配置将被存储在 `configs/generated/`.

## 测试 & 预测

### 手动下载：

请下载[SegFormer repository](https://github.com/NVlabs/SegFormer)官方库提供的在 ImageNet-1K 上预训练的 MiT 权重，并将它们放在本项目中的`pretrained/`文件夹中。大多数实验仅需要mit_b5.pth。

请从该[链接](https://drive.google.com/file/d/1mw8mTui-I-mvs2vo0UN_xs_fQh7AMpHm/view?usp=sharing)下载DIDA在 GTA->Cityscapes数据集上的权重参数并将其解压到`work_dirs/`

我们提供的在GTA→Cityscapes上训练的DIDA权重参数可以使用以下命令在Cityscapes验证集上测试：

```shell
sh test.sh work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa
```

预测结果保存在 `work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/preds`以供监督， 模型的 mIoU 打印在 `pre_log.out` 中。

我们提供的模型权重应当能达到 71.03 的 mIoU。 请查阅  `work_dirs/DIDA_gta2cs/221213_1011_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6e7fa/20221213_101429.log`的结尾位置以查看更多信息，例如按类的IoU。

与上述过程相似，其他模型也可以在训练后进行测试:

```shell
sh test.sh path/to/checkpoint_directory
```


在评估Synthia→Cityscapes上训练的模型时，请您注意：测试脚本计算了全部Cityscapes类（共19个）。但Synthia仅包含这些标签中的**16** 个。因此，UDA相关工作中的常规操作是对于Synthia→Cityscapes数据集仅报告这16个类上的mIoU。因为缺失的三个类的值为0，因此您可以做以下转化：**mIoU16 = mIoU19 * 19 / 16**。

预测结果可以提交到对应数据集的公开评估服务器以获得测试分数。

## 模型参数

下面，我们针对不同的基准提供了DIDA的模型参数。
由于论文中的结果是基于三个随机种子的平均值提供的，我们在此提供了具有**中位数验证性能**的权重。

* [DIDA for GTA→Cityscapes](https://drive.google.com/file/d/1mw8mTui-I-mvs2vo0UN_xs_fQh7AMpHm/view?usp=sharing)
* [DIDA for Synthia→Cityscapes](https://drive.google.com/file/d/1fpAsxbhlIxzPhiIfm0r8rAoTjK4PZb_V/view?usp=share_link)

这些模型参数均附有训练记录，请注意：

* 训练记录提供的mIoU是针对19个类的. 对于 Synthia→Cityscapes 数据集的 mIoU 有必要将其转化为针对16个有效类的值。具体操作请您阅读前文提到的转换mIoU的小节。


## 致谢

本项目参考了以下开源项目，我们由衷的感谢这些项目的作者公开了可用的源码。

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
* [DAFormer](https://github.com/lhoyer/DAFormer)

## 发布许可

该项目在[Apache License 2.0](LICENSE)下发布，而此库中的某些特定功能使用了其他许可。如果您将我们的代码商用，请仔细阅读 [LICENSES.md](LICENSES.md) 。

