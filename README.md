# DeepLearning-Techniques

Recall the important informations, not the details. 

Summary of visual recognition tasks:

- object detection
- segmentation (Semantic / Instance segmentation)
- action recoginition
- human-object interaction
- Autoencoder / Variational Autoencoder

The rule of neural networks

- It is a repreesntor for the probability distributions over the data.
- It is a automatic feature extractors using convoluitons.

## Old Models

**LeNet-5** First network by [LeCun](http://yann.lecun.com/exdb/lenet/). 

**AlexNet** Similar to LeNet but famous and provide a template for future CNN. 1) Used ImageSet; 2) Used GPU; 2) Lots of ideas (ReLU, Nomalization LRN, data argumentation, dropout).

**VGG16** Use small conv kernel to reduce the number of parameters. Parameters are still large in the last FC layer.

**Network-In-Network** Power for non-linear. 

**Inception** add 1*1 conv. Less parameters, combine different filter sizes (1x1, 3x3, 5x5), but parallel programming need more computer power.

**ResNet** Can be even deeper. Some layers are redundant. A followed up **ResNeXt** model which has similar ideas as Inception, Xeption, use depthwise convolution to decrease the number of parameters.

**DenseNet** CVPR 2017. connect layers with all other following layers. Inplement the idea with "Dense Block". Parameter small, but to many memory.

**R-CNN** [1311.2524](https://arxiv.org/abs/1311.2524) Use [Selective Search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf) method to propose 2000-3000 regions. Object detection in a larger size image, but also avoid to classify a huge number of regions. 

**Fast R-CNN** [1504.08083](https://arxiv.org/abs/1504.08083) Spatial Pyramid Pooling, ROI Pooling. You don’t have to feed 2000 region proposals each time. Only extract features one time.

**Faster R-CNN** [1506.01497](https://arxiv.org/pdf/1506.01497) Use Region Proposal Network (RPN) and anchors to propose boxes/regions.

**FCN** [CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) idea of upsample

**U-Net** [1505.04597](https://arxiv.org/abs/1505.04597) Combine
earlier higher resolution features and upsampled feature to increase get better representation. 

**SegNet** [1511.00561](https://arxiv.org/abs/1511.00561) no full connection layer, recording max-pooling indices, memory efficient. FCN, Deeplab, DeconvNet are compared to the model in the paper.

## Models
**YOLO**	<https://pjreddie.com/darknet/yolo/> [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

**YOLO2** [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)   BN, higher resolution, anchor boxes, Dimension Clusters to find anchor boxes, Multi-Scale Training

**YOLO3** [How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/), [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

**SSD** fast, but not good as YOLO? different scale detection.

**Mask R-CNN** [Marr Prize at ICCV 2017](https://arxiv.org/abs/1703.06870)

**RetinaNet** [Best Student Paper Award at ICCV 2017](https://arxiv.org/abs/1708.02002)

**DeepLab** 

## Other models
**RBM** restricted Boltzmann machine
**DBM** deep Boltzmann machine
**DBN** deep belief networks

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

**Imbalance** Only part of the image contains the target. Some classes have more complex / higher level features.

## Why Deep Learning Works in CV
* Translation invariant, scale invariant, Distortion invariant. 卷积网络结合了三种关键性思想来确保模型对图像的平移、缩放和扭曲具有一定程度的不变性，这三种关键思想即局部感受野、权重共享和空间/时间子采样 (ref: [机器之心](https://mp.weixin.qq.com/s/okx0jZR6PmFm3ikCCUbNkg)). Attention, CNN is not totally translation invariant as discussed in this paper [Why do deep convolutional networks generalize so poorly to small image transformations?](https://arxiv.org/abs/1805.12177)

## visualization
* t-SNE, T-Distributed Stochastic Neighbouring Entities, [orignal paper](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

* [CAM](https://arxiv.org/abs/1610.02391), [grad-CAM](https://arxiv.org/abs/1610.02391)

## Traditional CV Techniques

Old CV method

```
preprocess -> feature extraction -> feature selection -> classifier selection 
```

Deep learning method

```
preprocess -> CNN
```

* canny edge detection ([opencv tutorial](https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html)). A very useful edge detection method and will provide good intuiations after you learn it.
* SIFT Keypoints and Descriptors. ([opencv introduction](https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html)) Scale-Invariant Feature Transform
* A review paper, ["A Performance Evaluation of Local Descriptors" CVPR 2003](http://www.ai.mit.edu/courses/6.891/handouts/mikolajczyk_cvpr2003.pdf), shows the local descriptors and helps you to understand how edge detectors work. These are                     hand-craft features which are similar to what CNN created.

![](./**.png)



## Chinese demo resources
TO be tested [浅入浅出TensorFlow 8 - 行人分割](https://blog.csdn.net/linolzhang/article/details/70306708) tensorflow with Mask-RCNN