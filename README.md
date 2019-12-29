# LRF-Net-implemented-in-mmdetection

mmdetection versions of "Learning Rich Features at High-Speed for Single-Shot Object Detection", ICCV, 2019

## Introduction

This is a translation version of the offical implemention [vaesl/LRF-Net](https://github.com/vaesl/LRF-Net). You can easily customize your own dataset with the help of mmdetection toolbox.

## Installation and How to use

Please follow the instructions of [mmdetection/docs/INSTALL.md](https://github.com/open-mmlab/mmdetection/blob/8118d76a4ff609d5826bb5d758aebdd092d03392/docs/INSTALL.md). Put my code into the corresponding folders of mmdetection.
The tutorial of mmdetection is available at [GETTING_STARTED](https://github.com/open-mmlab/mmdetection/blob/8118d76a4ff609d5826bb5d758aebdd092d03392/docs/GETTING_STARTED.md).

## Compare with the offical version

### Dateset support

Dataset | This Repository | Offical Version
:--:|:--:|:--:
coco | √ | √
Pascal VOC|√|×

## Performance

### Performance in PASCAL VOC

All the following scores are trained with VOC 07+12 trainval by 24 epoch and test in VOC07 test.

Models|Performance
:--:|:--:
SSD300|0.767
LRF300|0.797
SSD512|0.793
LRF512|0.818

# ### Performance in COCO

All of the following scores are from the official trained models which transformed into mmdetection version.  i.e. I only  change the keys of OrderedDict.

Models|Train DataSet |Test DataSet | This Repository 
:--:|:--:|:--:|:--:
LRF300|coco2014 trainval | coco2017 val|0.267
LRF512|coco2014 trainval | coco2017 val|0.31


