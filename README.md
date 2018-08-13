# DeepLearning-Techniques

I recall the important informations, not the details. 

## Old Models
**AlexNet** Lots of ideas. to be added. But too many parameters

**VGG16** Use small conv kernel to reduce the number of parameters. Parameters are still large in the last FC layer.

**Inception** add 1*1 conv. Less parameters, but parallel programming need more computer power.

**ResNet** Can be even deeper. Some layers are redundant.

**DenseNet** CVPR 2017. connect layers with all other following layers. Inplement the idea with "Dense Block". Parameter small, but to many memory.

**R-CNN** [1311.2524](https://arxiv.org/abs/1311.2524) Use [Selective Search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf) method to propose 2000-3000 regions. Object detection in a larger size image, but also avoid to classify a huge number of regions. 

**Fast R-CNN** [1504.08083](https://arxiv.org/abs/1504.08083) Spatial Pyramid Pooling, ROI Pooling. You don’t have to feed 2000 region proposals each time. Only extract features one time.

**Faster R-CNN** [1506.01497](https://arxiv.org/pdf/1506.01497) Use Region Proposal Network (RPN) and anchors to propose boxes/regions.

## Models
**YOLO**	<https://pjreddie.com/darknet/yolo/> [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

**YOLO2** [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)   BN, higher resolution, anchor boxes, Dimension Clusters to find anchor boxes, Multi-Scale Training

**YOLO3** [How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/), [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

**SSD** fast, but not good as YOLO? different scale detection.

**Mask RCNN** unkown

**RetinaNet** unkown

## Techniques

**Spatial Pyramid Pooling** [1406.4729](https://arxiv.org/abs/1406.4729) deal with different size inputs.

**Region of Interested Pooling** ROI pooling. share features by combine the bbox regression and CNN.

**Region Proposal Network**

**Global Average Pooling** Object localization. [1312.4400.pdf](https://arxiv.org/pdf/1312.4400.pdf)

**Focal Loss** Use modulating factor to change the increase the weight for hard classes. Minor improvement. [Investigating Focal and Dice Loss for the Kaggle 2018 Data Science Bowl](https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c)

**softmax vs sigmoid** Softmaxing class scores assume that the classes are mutually exclusive.

**Edge Boxes** Locating Object Proposals from Edges, only use CV method.

**Selective Search** Region proposal method used in RCNN

## My tips

Rules of Thumb

```
- Create clean training data with good labels.
- Define the problem simply using CV method.
- Make the model robust using data argumentations.
- Choose the right task and right CNN (classification/segmentation/multi-scale detetion, and so on).
```

* Top tips
	* Check data and labels to make sure they are correct.
	* Check the code by overfitting a small dataset to make sure they are correct.
	* Get a baseline model as quick as possible.
	* Analyze the FP and FN, also the fake images.

* Simplyfy images features. 
	* Too much noise in the image, resize may help.
	* Use mask to focus on interested region
	
* Normalization
	* Use correct color region as normalization,

* Training
	* before a large run, please test it out. 
	* be clear why you run each task. 
	* analysis the task after each run.
	* Do not use large dropout in the begining to save time.
	
* Batch normalization tips
	* make sure you do not use bias term in the last conv layer.
	* a large batch size may slow down the training
	* a large batch size should be used with larger learning rate (> 1e-3). some references:[1609.04836](https://arxiv.org/pdf/1609.04836), [1711.00489](https://arxiv.org/abs/1711.00489)
	
### team works are hard in most case
- data
	- a
- traning
	- a

## Problems in DL

**Imbalance** Only part of the image contains the target.

## Why Deep Learning Works in CV
* Translation invariant, scale invariant, Distortion invariant. 卷积网络结合了三种关键性思想来确保模型对图像的平移、缩放和扭曲具有一定程度的不变性，这三种关键思想即局部感受野、权重共享和空间/时间子采样 (ref: [机器之心](https://mp.weixin.qq.com/s/okx0jZR6PmFm3ikCCUbNkg))。

## visualization
* t-SNE, T-Distributed Stochastic Neighbouring Entities, [orignal paper](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

*


![](./**.png)