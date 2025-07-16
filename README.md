<div align="center">

# CLRNet: Cross Layer Refinement Network for Lane Detection

</div>



Pytorch implementation of the paper "[CLRNet: Cross Layer Refinement Network for Lane Detection](https://arxiv.org/abs/2203.10350)" (CVPR2022 Acceptance).  
Adapted for use within a Hochschule Esslingen study project [project report](Report.md)

## Introduction
![Arch](.github/arch.png)
- CLRNet exploits more contextual information to detect lanes while leveraging local detailed lane features to improve localization accuracy. 
- CLRNet achieves SOTA result on CULane, Tusimple, and LLAMAS datasets.

## Installation

### Prerequisites
Tested on Ubuntu 22.04.2 with:
- Python >= 3.8 (tested with Python3.8.20)
- PyTorch >= 1.8 (tested with Pytorch1.8)
- CUDA (tested with cuda12.1)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$CLRNET_ROOT`
```Shell
git clone https://github.com/vdw91/HSE_sim_CLRNet.git
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n clrnet python=3.8 -y
conda activate clrnet
```

### Install dependencies

```Shell
# Use pip to install all packages
pip install -r requirements.txt

# Create build environment and packages
python setup.py build develop
```

Note: This gives an error message, but should be ok: 
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
pypcd 0.1.3 requires python-lzf, which is not installed.
requests-oauthlib 2.0.0 requires oauthlib>=3.0.0, which is not installed.
tensorflow 2.13.1 requires typing-extensions<4.6.0,>=3.6.6, but you have typing-extensions 4.13.2 which is incompatible.
```

## Labelling
The repository comes with a set of tools which were used to manually label the SimSimple Dataset.

To unpack the data set, run 
```bash
cd datasets/
sudo apt install p7zip-full
7za x simsimple.7z 
```

#### Manual labelling

To start the manual labeller, run
```Shell
python labelling_tools/manual_labeller.py [path_to_dir_with_images] [path_to_dir_where_labels_will_be_saved]
```
The labelling tool should work with png images of size 800x800.

The manual labeller labels lanes 0 through 2. (Currently no option to go back to a previous lane)  
To conform to the TuSimple dataset, start by labelling the middle lane (0).
To do so, start by selecting click within the lane, next click on another point within the lane. The tool will draw a line between the points and store it. To continue drawing the lane simply click again and the line will continue.  
If the connecting point is incorrect, use backspace, which removes the last drawn line. (but keeps the starting point)  
If the starting point needs to be changed, press backspace and then delete.  
To continue labelling the next lane (which is the lane to the right of the vehicle), press the right mouse button.
Repeat this for the next and final lane.  
Once the image has been properly labelled, press enter to continue to the next image.


#### Quality assurance

To start the quality assurance tool, run
```Shell
python labelling_tools/quality_checker.py [path_to_dir_where_manual_labeller_output]
```

Once the quality assurance tool has been opened a white screen will be displayed. Press enter to view the first image.
The tool will display the image along with the labels. If the labels are unsatisfactory, press delete. This will delete the image and the label from the output dir (the directory given to the tool when called), which lets the manual labeller label it again.  
If the labels are correct, press enter, which keeps the image and the labels.

#### Restructure manual labeller output

To start restructuring tool, run
```Shell
python labelling_tools/restructure_labels_to_dataset.py [path_where_manual_labeller_saved_the_data] [path where dataset images will be saved] [Path where dataset JSON files will be saved]
```

The tool restructures the output from the manual labeller, which prioritises human readability, to the format desired by model.


### Data preparation
### SimSimple
The [SimSimple](https://github.com/vdw91/HSE_sim_CLRNet/blob/main/datasets/simsimple.7z) dataset is contained as a 7z file within the repository. Then extract them to `$SIMSIMPLEROOT`. Create link to `data` directory.

```Shell
export CLRNET_ROOT=[your data directory]
cd $CLRNET_ROOT
mkdir -p data
export SIMSIMPLEROOT=$CLRNET_ROOT/data/simsimple
.. unzip simsimple.7z in $SIMSIMPLEROOT ..
```

For SimSimple, you should have structure like this:
```
$SIMSIMPLEROOT/clips # data folders
$SIMSIMPLEROOT/seg_label # segmentation labels
$SIMSIMPLEROOT/train_set.json # train labels
$SIMSIMPLEROOT/test_set.json # test labels
$SIMSIMPLEROOT/val_set.json # validation labels
```

The SimSimple dataset will come with the segmentation labels. If you decide to add new clips, segementation labels need to be generated before training can be started:

```Shell
python tools/generate_seg_tusimple.py --root $SIMSIMPLEROOT --simsimple
# this will generate seg_label directory
```

#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```


## Getting Started

### Training
For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/clrnet/clr_resnet18_simsimple.py --gpus 0
```

Currently the live training progress can only be checked in the terminal. Since all the logs are saved a tool is provided to display these logs as a series of graphs. 
To see the training graphs, run 
```Shell
python tools/plot_training.py [path_to_training_output.txt]
```

For example, run
```Shell
python tools/plot_training.py trained_models/resnet_34/log.txt
```

To load our training result, you can download the released model from github:
```bash
mkdir clrnet_model # somewhere you like
cd clrnet_model/
wget https://github.com/vdw91/HSE_sim_CLRNet/releases/download/models/trained_models.zip
unzip trained_models.zip 
```

### Validation
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/clrnet/clr_resnet18_simsimple.py --test --load_from trained_models/resnet_34/ckpt/resnet_34_simsimple.pth --gpus 0
```

### Visualisation
To visualise the models performance a visualisation tool is offered, which displays the prediction on a single image, or a series of images if a directory is provided.

For visualisation, run
```Shell
python visualise.py [configs/path_to_your_config] --img [path_to_single_image or path_to_folder_of_images]  --load_from [path_to_your_model] --show
```

For example, run
```Shell
python visualise.py configs/clrnet/clr_resnet18_simsimple.py --img img.png  --load_from trained_models/resnet_34/ckpt/resnet_34_simsimple.pth --show
```



## Results
![F1 vs. Latency for SOTA methods on the lane detection](.github/latency_f1score.png)

[assets]: https://github.com/turoad/CLRNet/releases
[models]: https://github.com/vdw91/HSE_sim_CLRNet/releases/tag/models

### SimSimple
|   Backbone   |      F1   | Acc |      FDR     |      FNR   |
|    :---       |          ---:          |       ---:       |       ---:       |      ---:       |
| [ResNet-18][models]     |    97.37    |   94.50  |    2.63  |  2.63      | 
| [ResNet-34][models]       |   98.25              |    96.47          |   1.75          |    1.75      | 
| [ResNet-101][models]      |   98.25|   96.64  |   1.75   |  1.75  |



### TuSimple
|   Backbone   |      F1   | Acc |      FDR     |      FNR   |
|    :---       |          ---:          |       ---:       |       ---:       |      ---:       |
| [ResNet-18][assets]     |    97.89    |   96.84  |    2.28  |  1.92      | 
| [ResNet-34][assets]       |   97.82              |    96.87          |   2.27          |    2.08      | 
| [ResNet-101][assets]      |   97.62|   96.83  |   2.37   |  2.38  |



### CULane

|   Backbone  |  mF1 | F1@50  | F1@75 |
| :---  |  :---:   |   :---:    | :---:|
| [ResNet-18][assets]     | 55.23  |  79.58   | 62.21 |
| [ResNet-34][assets]     | 55.14  |  79.73   | 62.11 |
| [ResNet-101][assets]     | 55.55| 80.13   | 62.96 |
| [DLA-34][assets]     | 55.64|  80.47   | 62.78 |



### LLAMAS
|   Backbone    |  <center>  valid <br><center> &nbsp; mF1 &nbsp; &nbsp;  &nbsp;F1@50 &nbsp; F1@75     | <center>  test <br> F1@50 |
|  :---:  |    :---:    |        :---:|
| [ResNet-18][assets] |  <center> 70.83  &nbsp; &nbsp; 96.93 &nbsp; &nbsp; 85.23 | 96.00 |
| [DLA-34][assets]     |  <center> 71.57 &nbsp; &nbsp;  97.06  &nbsp; &nbsp; 85.43  |   96.12 |

“F1@50” refers to the official metric, i.e., F1 score when IoU threshold is 0.5 between the gt and prediction. "F1@75" is the F1 score when IoU threshold is 0.75.

## Citation

If our paper and code are beneficial to your work, please consider citing:
```
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
    title     = {CLRNet: Cross Layer Refinement Network for Lane Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {898-907}
}
```

## Acknowledgement
<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->