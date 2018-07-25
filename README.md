# DeepLearning-Techniques

## Old Models
**AlexNet** Lots of ideas. to be added. But too many parameters

**VGG16** Use small conv kernel to reduce the number of parameters. Parameters are still large in the last FC layer.

**Inception** add 1*1 conv. Less parameters, but parallel programming need more computer power.

**ResNet** Can be even deeper. Some layers are redundant.

**DenseNet** CVPR 2017. connect layers with all other following layers. Inplement the idea with "Dense Block". Parameter small, but to many memory.

## Models
**YOLO**	<https://pjreddie.com/darknet/yolo/>

**R-CNN** Object detection in a larger size image.

**Fast R-CNN** Spatial Pyramid Pooling, ROI Pooling

**Faster R-CNN** Use Region Proposal Network to propose boxes/regions.

**SSD** fast, but not good as YOLO

## Techniques

**Spatial Pyramid Pooling** deal with different size inputs

**Region of Interested Pooling** ROI pooling. share features by combine the bbox regression and CNN.

**Region Proposal Network**

**Global Average Pooling** Object localization. [1312.4400.pdf](https://arxiv.org/pdf/1312.4400.pdf)

**Focal Loss** Use modulating factor to change the increase the weight for hard classes. Minor improvement. [Investigating Focal and Dice Loss for the Kaggle 2018 Data Science Bowl](https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c)

## Problems in DL

**Imbalance** Only part of the image contains the target.

## Why Deep Learning Works in CV
* Translation invariant, scale invariant, Distortion invariant. 卷积网络结合了三种关键性思想来确保模型对图像的平移、缩放和扭曲具有一定程度的不变性，这三种关键思想即局部感受野、权重共享和空间/时间子采样 (ref: [机器之心](https://mp.weixin.qq.com/s/okx0jZR6PmFm3ikCCUbNkg))。


![](./**.png)