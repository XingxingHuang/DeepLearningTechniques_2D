# DeepLearning-Techniques

This repo is for recalling the important informations, not for the details. So I will not explain too much in each topic. But I am thinking to posting more thoughts in to [https://medium.com] in 2019. 

Summary of visual recognition tasks:

- object detection
- object classification
- object localization
- segmentation (Semantic / Instance segmentation)
- Autoencoder / Variational Autoencoder
- action recoginition
- human-object interaction

The rule of neural networks

- It is a repreesntor for the probability distributions over the data.
- It is a automatic feature extractors using convoluitons.

## Models

**LeNet-5** First network by [LeCun](http://yann.lecun.com/exdb/lenet/). 

**AlexNet** Similar to LeNet but famous and provide a template for future CNN. 1) Used ImageSet; 2) Used GPU; 2) Lots of ideas (ReLU, Nomalization LRN, data argumentation, dropout).

**OverFeat** [1312.6229](https://arxiv.org/pdf/1312.6229) combine localization, localization, detection. The winner of localization task in ILSVR2013

**VGG16** Use small conv kernel to reduce the number of parameters. Parameters are still large in the last FC layer.

**Network-In-Network** Power for non-linear. 

**Inception** add 1*1 conv. Less parameters, combine different filter sizes (1x1, 3x3, 5x5), but parallel programming need more computer power. In the paper, the author says he inspired by the [Sparsely connected architectures](https://arxiv.org/abs/1310.6343). While the original paper is hard to read, the principle, Hebbian principle,  is very interesting. The follow up papers are [Inception v2 and v3](https://arxiv.org/pdf/1512.00567v3.pdf) and [Inception v4, Inception-ResNet](https://arxiv.org/pdf/1602.07261.pdf). A good comparision is found in [Medium](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) (A Simple Guide to the Versions of the Inception Network).

**ResNet** Can be even deeper. Some layers are redundant. A followed up **ResNeXt** model which has similar ideas as Inception, Xeption, use depthwise convolution to decrease the number of parameters.

**DenseNet** CVPR 2017. connect layers with all other following layers. Inplement the idea with "Dense Block". Parameter small, but to many memory.

**ResNeXt** Briliant idea that combine resnet and Xception. Good paper to read.

**YOLO**	[1506.02640](https://arxiv.org/abs/1506.02640) Provide an end to end solution to detect multiple objects in one images. The model predicts the location and the class in the mean time (5 parameters). <https://pjreddie.com/darknet/yolo/> [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

**YOLO2** [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)   BN, higher resolution, anchor boxes, Dimension Clusters to find anchor boxes, fine-gained feature (skip connection similar with ResNet), Multi-Scale Training (different input size instead of resize). Check the paper to check how many methods they have tried.

**YOLO3** [1804.02767](https://arxiv.org/abs/1804.02767). At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. Get better performance for small objects. [How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) Make prediction with 9 archor boxes and 3 scales. Use independent logistic classifiers instead of oen softmax. Create DarkNet based on Residual Net by adding some shortcut connections.

**SSD** [1512.02325](1512.02325) Combined ideas from R-CNN and YOLO. Provide different scale detection. Faster but slightly worse performance (Better performance compared to Faster RCNN is report in the paper).

**R-CNN** [1311.2524](https://arxiv.org/abs/1311.2524) R-CNN decomposes the overall detection problem into two subproblems: to first utilize low-level cues such as color and superpixel consistency for potential object proposals in a category-agnostic fashion, and to then use CNN classifiers to identify object categories at those locations. R-CNN use [Selective Search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf) method to propose 2000-3000 regions, CNN as feature extractor, SVM as classifier. Object detection in a larger size image, but also avoid to classify a huge number of regions. 

**Fast R-CNN** [1504.08083](https://arxiv.org/abs/1504.08083) You don’t have to feed 2000 region proposals each time. Only extract features one time. Used [Spatial Pyramid Pooling, 1406.4729](https://arxiv.org/abs/1406.4729) idea, proposed ROI Pooling to get fixed output size. Used SoftmaxLoss instead of SVM, SmoothL1Loss instead of Bounding box.

**Faster R-CNN** [1506.01497](https://arxiv.org/pdf/1506.01497) Use Region Proposal Network (RPN) and anchors to propose boxes/regions.

**Mask R-CNN** [Marr Prize at ICCV 2017](https://arxiv.org/abs/1703.06870) Extend faster R-CNN by adding a branch for predicting segmentation masks.

**FCN** [CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) idea of upsample

**U-Net** [1505.04597](https://arxiv.org/abs/1505.04597) Combine
earlier higher resolution features and upsampled feature to increase get better representation. 

**SegNet** [1511.00561](https://arxiv.org/abs/1511.00561) no full connection layer, recording max-pooling indices, memory efficient. FCN, Deeplab, DeconvNet are compared to this model in the paper.

**FCN** [1612.03144](https://arxiv.org/abs/1612.03144) Feature Pyramid Networks for Object Detection. They exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost.

**RetinaNet** [Best Student Paper Award at ICCV 2017](https://arxiv.org/abs/1708.02002) Use focal loss ( keep using FPN model). One stage detector, fast and accurate. But YOLO3 claim higher performance in the paper.

**DeepLab** 

## Other models

**RBM** restricted Boltzmann machine
**DBM** deep Boltzmann machine
**DBN** deep belief networks

## Techniques

**Spatial Pyramid Pooling** [1406.4729](https://arxiv.org/abs/1406.4729) Kaiming he. Also called SPP-Net. deal with different size inputs. Smart idea, but I think the use pooling for different scales may still cause problem. Especially your target has a large size variation. 

**SQueezeNet** [1602.07360](https://arxiv.org/abs/1602.07360) Use the Fire module to squeeze the networks and get compressed model ~ 0.5 MB. Found DSD (Dense→Sparse→Dense) method that to use spared pretrain mdoel to retrain could get better results.

**Region of Interested Pooling** ROI pooling. share features by combine the bbox regression and CNN.

**Global Average Pooling** Object localization. [1312.4400.pdf](https://arxiv.org/pdf/1312.4400.pdf)

**Focal Loss** Use modulating factor to change the increase the weight for hard classes. Minor improvement. [Investigating Focal and Dice Loss for the Kaggle 2018 Data Science Bowl](https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c)

**softmax vs sigmoid** Softmaxing class scores assume that the classes are mutually exclusive.

**Selective Search** [Selective Search for Object Recognition](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) A traditional CV method for region proposals used in RCNN. Considered color, texture, size, fill similarity between blocks and combine them using the minimum spanning tree method.

**Edge Boxes** Locating Object Proposals from Edges, use only CV method. decrease the region proposal speed from 2s (Selective Search) to 0.2 s

**RNN** RNN > LSTM > GRU > 

## Interesting Applications
**Neural Style Transfer** [Gatys et al., 2015a](https://arxiv.org/abs/1505.07376), [Gatys et al., 2015b](https://arxiv.org/abs/1508.06576)

## Tips

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

### Features

```
- Haar
- SIFT
- HOG
- convolutional features
```

* canny edge detection ([opencv tutorial](https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html)). A very useful edge detection method and will provide good intuiations after you learn it.
* SIFT Keypoints and Descriptors. ([opencv introduction](https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html)) Scale-Invariant Feature Transform
* A review paper, ["A Performance Evaluation of Local Descriptors" CVPR 2003](http://www.ai.mit.edu/courses/6.891/handouts/mikolajczyk_cvpr2003.pdf), shows the local descriptors and helps you to understand how edge detectors work. There are hand-craft features which are similar to what CNN created.

### Sigal processing

[Wiener Filter](https://en.wikipedia.org/wiki/Wiener_filter) (维纳滤波) a filter used to produce an estimate of a desired or target random process by linear time-invariant (LTI) filtering of an observed noisy process, assuming known stationary signal and noise spectra, and additive noise.

![](./**.png)


## Chinese demo resources
TO be tested [浅入浅出TensorFlow 8 - 行人分割](https://blog.csdn.net/linolzhang/article/details/70306708) tensorflow with Mask-RCNN