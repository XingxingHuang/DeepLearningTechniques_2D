# DeepLearning-Techniques

This repo is for the purpose of recalling some important information, not for explaining all details. So I will not expand each topic to long articals. I am planning to posting more thoughts of deep learning into [https://medium.com](https://medium.com) in 2019. 

## The cue of neural networks

- It is still a repreesntor for the probability distributions over the data.
- It is a automatic feature extractors using convoluitons.
- It cannot beat human level performance (bayes errors), but it still has its own advantage. (This is my idea. Glad to be contradictted.)


You can find the structure of different models in folder [images](./images).
You can find my collection of different learning curves in [learning_curves](./learning_curves/README.md). Believe me, you will learning a lot by just checking these beautiful curves.

Summary of visual recognition tasks (GAN not included):

- Image classification
- Object detection
- Object localization
- Segmentation (Semantic / Instance segmentation)
- Image matting (one pixel is a combination of forground and background)
- Autoencoder / Variational Autoencoder
- Action recoginition
- Human-object interaction


Current big problems

- Small object detection. Some objects are small while others are large. 	
- Imbalance. Only part of the image contains the target. Some classes have more complex / higher level features.

- To be added.

## Models

**LeNet-5** First network by [LeCun](http://yann.lecun.com/exdb/lenet/). LeNet-5 is a very simple network. It only has 7 layers, among which there are 3 convolutional layers (C1, C3 and C5), 2 sub-sampling (pooling) layers (S2 and S4), and 1 fully connected layer (F6), that are followed by the output layer. 

**AlexNet** [papers.nips.cc](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) Similar to LeNet but famous and provide a template for future CNN. 1) Used ImageSet; 2) Used GPU; 2) Lots of ideas (ReLU, Nomalization LRN (local response normalization), data argumentation, dropout, use max-pooling intead of average pooling). `LRN is found useless later as compared to BN and dropped afterward.`

**ZFNet** [1311.2901](https://arxiv.org/abs/1311.2901) Winner of 2013 ILSVR. Similar to AlexNet, uses 7x7 kernels instead of 11x11 kernels to significantly reduce the number of weights. Also introduces the visualization method with deconv.

**OverFeat** [1312.6229](https://arxiv.org/pdf/1312.6229) combine localization, localization, detection. The winner of localization task in ILSVR2013. Use multi-scale method for classification, combine classification network and regression network for localization.

**VGG16** [1409.1556](https://arxiv.org/abs/1409.1556) Great paper to read. Use small conv kernel to reduce the number of parameters and make the model deeper. Parameters in VGG models are still large in the last FC layer. They used 1x1 conv, but haven't notice the benifits.

**Network-In-Network** [1312.4400](https://arxiv.org/abs/1312.4400) Power for non-linear. Reduce the AlexNet from 230M to 29M. Introduced mlpconv layer and increased the nolinearity. Used a global average pooling layer instead of FC layer to reduce the overfit.

**Inception** add 1*1 conv. Less parameters, combine different filter sizes (1x1, 3x3, 5x5), but parallel programming need more computer power. They use two auxiliary classifiers to avoid gradient vanish. In the paper, the author says he inspired by the [Sparsely connected architectures, 1310.6343](https://arxiv.org/abs/1310.6343). While the original paper is hard to read, the principle, [Hebbian principle](https://en.wikipedia.org/wiki/Hebbian_theory),  is very interesting. You can simplify the idea as `neurons that fire together, wire together`. The follow-up papers are [Inception v2 and v3](https://arxiv.org/pdf/1512.00567v3.pdf) and [Inception v4, Inception-ResNet](https://arxiv.org/pdf/1602.07261.pdf). A good comparision is found in [Medium](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) (A Simple Guide to the Versions of the Inception Network).

- v2 and v3. use two 3x3 conv instead of 5x5 conv. Use 1xn and nx1 intead of nxn conv. Expand the filter bank. Use label smoothing for the regularization.
- v4. Improve the "stem", learn the idea from Resnet.

**ResNet** [1512.03385](https://arxiv.org/abs/1512.03385) Can be even deeper. Some layers are redundant, so shortcut connections help to reduce the model complexity and training. A followed up **ResNeXt** model which has similar ideas as Inception, Xeption, use depthwise convolution to decrease the number of parameters.

**DenseNet** [1608.06993](https://arxiv.org/abs/1608.06993) CVPR 2017. connect layers with all other following layers. Inplement the idea with "Dense Block". Parameter small, but to many memory (further discussion about this in [1707.06990](https://arxiv.org/abs/1707.06990)). Use Transition module to connect denseblocks.

**ResNeXt** [1611.05431](https://arxiv.org/abs/1611.05431) Good paper to read. Briliant idea that combine resnet and Xception. Mentioned the "cardinality" as the size of the set of transformations. The experiments demostrate that increasing the cardinality is a more effective way of gaining accuracy than going deeper or wider. 

**YOLO**	[1506.02640](https://arxiv.org/abs/1506.02640) Provide an end to end solution to detect multiple objects in one images. The model predicts the location and the class in the mean time (5 parameters). Use Non-maximum Suppression
 to cure the problem of multiple detections of the same image. <https://pjreddie.com/darknet/yolo/> 
```
grid: 7x7
box: 2
class: N
Loss function: box coordinate + box size (square) + confidence
```

**YOLO2** [YOLO9000: Better, Faster, Stronger, 1612.08242](https://arxiv.org/abs/1612.08242) Great paper to read. They tried many methods to improve the model including BN, higher resolution, anchor boxes, Dimension Clusters to find anchor boxes, fine-gained feature (skip connection similar with ResNet), Multi-Scale Training (different input size instead of resize, you can train the model with different sizes). Check the paper to learn how many methods they have tried.

**YOLO3** [1804.02767](https://arxiv.org/abs/1804.02767). At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. Get better performance for small objects. [How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) Make prediction with 9 archor boxes for 3 scales (3 boxes for 3 anchor boxes, each scale has different anchor boxes). Use independent logistic classifiers instead of the softmax (Softmaxing classes rests on the assumption that classes are mutually exclusive). Create DarkNet based on Residual Net by adding some shortcut connections and extract features from 3 different scale.

**SSD** [1512.02325](https://arxiv.org/abs/1512.02325) Combined ideas from R-CNN and YOLO. Provide different scale detection. Compared to YOLOv3, SSD uses Softmax loss, uses VGG19 model, has Different scale and aspect ratio (5 values in the paper) for the anchor boxes, needs resize original image to fixed size. Faster but slightly worse performance (Better performance compared to Faster RCNN is report in the paper).

**R-CNN** [1311.2524](https://arxiv.org/abs/1311.2524) R-CNN decomposes the overall detection problem into two subproblems: to first utilize low-level cues such as color and superpixel consistency for potential object proposals in a category-agnostic fashion, and to then use CNN classifiers to identify object categories at those locations. R-CNN use [Selective Search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf) method to propose 2000-3000 regions, CNN as feature extractor, SVM as classifier. Object detection in a larger size image, but also avoid to classify a huge number of regions. 

**Fast R-CNN** [1504.08083](https://arxiv.org/abs/1504.08083) You don’t have to feed 2000 region proposals each time. Only extract features one time. Used Spatial Pyramid Pooling [1406.4729](https://arxiv.org/abs/1406.4729) idea, proposed ROI Pooling to get fixed output size. Used SoftmaxLoss instead of SVM, SmoothL1Loss instead of Bounding box. Share parameters between box regressions and classifications.

**Faster R-CNN** [1506.01497](https://arxiv.org/pdf/1506.01497) Extract feature map and then use Region Proposal Network (RPN) and anchors to propose boxes/regions. They only use 9 anchors with 3 different scales and 3 aspect ratios.

**Mask R-CNN** [Marr Prize at ICCV 2017](https://arxiv.org/abs/1703.06870) A Great paper to read. "Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each in- stance." It extends faster R-CNN by adding a branch for predicting segmentation masks. (x) Use ResNeXt-101 + FPN for feature extraction. (x) replace the ROI pooling to ROI align (solved mis-alignment problem with ROI pooling). (x) add FCN lay ers to get segmentations. This model is not huge improvement, but it can get state of art performance and be easily extend to other tasks. Check [the oral slide](https://www.slideshare.net/windmdk/mask-rcnn) by Kaiming in ICCV 2017 to get the R-CNN evolution.

```
You can image that using archor boxes in these object detection algorithm is just using human prior to simplify the problems. 
How to design the optimal distributions of boxes is an open question as said in SSD paper. 
I believe it is important to take human prior to solve problems in your industrial projects. 
Invloving these priors can simplify the problems and increase the model robustness.
You can take more human priors into acount by thinking how to format the problems.
```
**R-FCN** [NIPS2016, 1605.06409](https://arxiv.org/abs/1605.06409) with github [code](https://github.com/daijifeng001/r-fcn). By positive sensitive score map, the inference time is much faster than Faster R-CNN while still maintaining competitive accuracy. In R-CNN, the process (FC layers) after ROI pooling does not share among ROI, and takes time, which makes RPN approaches slow. And the FC layers increase the number of connections (parameters) and the complexity. In R-FCN, FC layers after ROI pooling are removed. Instead, all major complexity is moved before ROI pooling to generate the score maps. 

**FCN** [CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) idea of upsample and provide pixel level predictions. You can create FCN-32s,FCN-16s,FCN-8s with different upsample sizes.

**InstanceFCN** [1603.08678](https://arxiv.org/abs/1603.08678) Using InstanceFCN, each score map is responsible for capturing relative position of object instance. A very nice introduction from [towardsdatascience](https://towardsdatascience.com/review-instancefcn-instance-sensitive-score-maps-instance-segmentation-dbfe67d4ee92). R-FCN uses positive-sensitive score maps for object detection, while InstanceFCN uses instance-sensitive score maps for generating proposals during the instance-sensitive score maps branch. During objectness score map branch, it uses a 1x1 layer as a per-pixel logistic regression for classifying instance/not-instance of the sliding window centered at this pixel.

**U-Net** [1505.04597](https://arxiv.org/abs/1505.04597) Combine earlier higher resolution features and upsampled feature to increase get better representation. 

**SegNet** [1511.00561](https://arxiv.org/abs/1511.00561) SegNet includes encoder network, decoder network, pixel-wise classification layer. no full connection layer, recording max-pooling indices, memory efficient. FCN, Deeplab, DeconvNet (but nod U-Net) are compared in this paper. The significant contribution is that the maxpooling indices transferred to decoder to improve the segmentation resolution.

**FPN** [1612.03144](https://arxiv.org/abs/1612.03144) Feature Pyramid Networks for Object Detection. They exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. In past networks, Fast RCNN, Faster RCNN only use the last layers as feature map, SSD uses different layers but doesn't have upsample and combine different scales. 

**RetinaNet** Best Student Paper Award at ICCV 2017, [1708.02002](https://arxiv.org/abs/1708.02002) Use focal loss (still using FPN model) to solve the class unbalance problem (especially most regions are background. This problem is reduced a little in two stage models as the model will distinguish backgrounds or objects before the classification.). And the author design a signle stage model RetinaNet to check how useful the focal loss is. `RetinaNet = FPN + ResNet + FL`.  RetinaNet is one stage detector, fast and accurate. Although RetinaNet says higher performance compared to YOLOv2, YOLOv3 claims higher performance in their paper.

--

#### Haven't finish the following series.

[Deeply-Supervised Nets, 1409.5185](https://arxiv.org/abs/1409.5185)

**DeepMask** [towardsdatascience.com](https://towardsdatascience.com/review-deepmask-instance-segmentation-30327a072339)

**RefineNet** [1611.06612](https://arxiv.org/abs/1611.06612)

**PSPNet** [1612.01105](https://arxiv.org/abs/1612.01105)

**MobileNet** 

**ShuffleNet**

**FractalNet** [1605.07648](https://arxiv.org/abs/1605.07648)

**DeepLab** [1606.00915](https://arxiv.org/abs/1606.00915) The implementation with Tensoflow can be found in Github/tensorflow/models/research/[deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)

DeepLabv1 [1412.7062](https://arxiv.org/abs/1412.7062): Use atrous convolution to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks.

DeepLabv2 [1606.00915](https://arxiv.org/abs/1606.00915): Use atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales with filters at multiple sampling rates and effective fields-of-views.

DeepLabv3 [1706.05587](https://arxiv.org/abs/1706.05587): Augment the ASPP module with image-level feature [5, 6] to capture longer range information. We also include batch normalization [7] parameters to facilitate the training. In particular, we applying atrous convolution to extract output features at different output strides during training and evaluation, which efficiently enables training BN at output stride = 16 and attains a high performance at output stride = 8 during evaluation.

DeepLabv3+ [1802.02611](https://arxiv.org/abs/1802.02611): Extend DeepLabv3 to include a simple yet effective decoder module to refine the segmentation results especially along object boundaries. Furthermore, in this encoder-decoder structure one can arbitrarily control the resolution of extracted encoder features by atrous convolution to trade-off precision and runtime.

## Other models

**RBM** restricted Boltzmann machine
**DBM** deep Boltzmann machine
**DBN** deep belief networks

**FaceNet**

**FCNT** **GOTURN** **C-COT** **SiameseFC** object tracking

**CRNN** **CTPN** OCR

**Semantic Human Matting** [1809.01354](https://arxiv.org/abs/1809.01354) Collect a large dataset for the image matting problem. Designed an end to end solution with three part: T-Net, which is any kinds of segmentation model to get three map for forground, background, unkown region segmentation (called trimaps); M-Net, which is used to get the alpha channel with RGB images and output from T-Net; Fusion module to get the alpha map with the output from above two networks. Check a related work [alphaGAN](https://neurohive.io/en/state-of-the-art/alphagan-natural-image-matting/) which also shows the related works in this field.


## Techniques

**Batch Normalization** [1805.11604v3](https://arxiv.org/abs/1805.11604) This paper talks about why BN helps. The popular belief is that the BN reduces the so-called “internal covariate shift”. The paper demonstrates that such distributional stability of layer inputs has little to do with the success of BatchNorm. Instead, a more fundamental impact of BatchNorm on the training process: it makes the optimization landscape significantly smoother. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training.

**ReLu, Leaky Relu, ELU, SELU** The ReLu keep the gradient to 1 to avoid the gradient vanishing and exploding problems, but it face dying ReLu problem as the model has low response for negative values. The [Leaky ReLu](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) in 2013, [ELU](https://arxiv.org/abs/1511.07289) in 2015, [SELU](https://arxiv.org/abs/1706.02515) (scaled exponential Linear Unit) in 2017 solves the problem by change the format when x < 0.

**astrous/dilated conv** [1511.07122](https://arxiv.org/abs/1511.07122) [DeepLab, 1606.00915](https://arxiv.org/abs/1606.00915) invented a convlution strategy to improve the performance for different object sizes.

**Spatial Pyramid Pooling** [1406.4729](https://arxiv.org/abs/1406.4729) Kaiming he. Also called SPP-Net. deal with different size inputs. Smart idea, but I think the use pooling for different scales may still cause problem. Especially your target has a large size variation. 

**SQueezeNet** [1602.07360](https://arxiv.org/abs/1602.07360) Use the Fire module to squeeze the networks and get compressed model ~ 0.5 MB. Found DSD (Dense→Sparse→Dense) method that to use spared pretrain mdoel to retrain could get better results.

**Region of Interested Pooling** ROI pooling. share features by combine the bbox regression and CNN.

**Global Average Pooling** Object localization. [1312.4400.pdf](https://arxiv.org/pdf/1312.4400.pdf)

**Focal Loss** Use modulating factor to change the increase the weight for hard classes. Minor improvement. [Investigating Focal and Dice Loss for the Kaggle 2018 Data Science Bowl](https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c)

**softmax vs sigmoid** Softmaxing class scores assume that the classes are mutually exclusive.

**Selective Search** [Selective Search for Object Recognition](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) A traditional CV method for region proposals used in RCNN. Considered color, texture, size, fill similarity between blocks and combine them using the minimum spanning tree method.

**Edge Boxes** Locating Object Proposals from Edges, use only CV method. decrease the region proposal speed from 2s (Selective Search) to 0.2 s

**Group Normalization** [1803.08494](https://arxiv.org/abs/1803.08494) GN divides the channels into groups and computes within each group the mean and variance for normalization. GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes.

**CRF** Conditional Random Field (CRF) postprocessing are usually used to improve the segmentation. CRFs are graphical models which ‘smooth’ segmentation based on the underlying image intensities. They work based on the observation that similar intensity pixels tend to be labeled as the same class. CRFs can boost scores by 1-2%.

**RNN** RNN > LSTM > GRU > 

## Interesting Applications
**Neural Style Transfer** [Gatys et al., 2015a](https://arxiv.org/abs/1505.07376), [Gatys et al., 2015b](https://arxiv.org/abs/1508.06576)

To be added

## Tips

Rules of Thumb

```
- Create clean training data (!) with clean (!) labels.
- Solve the problem with deep learning and (!) CV method.
- Make the model robust using data argumentations or preprocesses.
- Choose the right task and right CNN (classification/segmentation/multi-scale detetion, and so on). This is very tricky as sometime times people want you to detect xxx, but actually you can think the problem in other ways like segmentation/ abnormal detection / etc.
```

* Top tips
	* Check data and labels to make sure they are correct.
	* Check the code by overfitting a small dataset to make sure they are correct.
	* Get a baseline model as quick as possible.
	* Analyze the FP and FN, also check with fake images.

* Simplify images features. 
	* Only focus on the region of interest 
	* Too much noise in the image, preprocesses such as resize may help.
	* Use mask to focus on interested region
	
* Image processing
	* Use proper regions of interest to normalize the images.
	* Before training, carefully check your processing methods!

* Training and iteration
	* before a large run, please test it out. 
	* be clear why you run each task. 
	* analysis the task after each run.
	* Do not use large dropout in the begining to save time.
	
* Batch normalization tips
	* make sure you do not use bias term in the last conv layer.
	* a large batch size may slow down the training
	* a large batch size should be used with larger learning rate (> 1e-3). some references:[1609.04836](https://arxiv.org/pdf/1609.04836), [1711.00489](https://arxiv.org/abs/1711.00489)
	
### team works are hard in most case
- organize the data
- organize the training
- discuss the results

## Why Deep Learning Works in CV
* Translation invariant, scale invariant, Distortion invariant. 卷积网络结合了三种关键性思想来确保模型对图像的平移、缩放和扭曲具有一定程度的不变性，这三种关键思想即局部感受野、权重共享和空间/时间子采样 (ref: [机器之心](https://mp.weixin.qq.com/s/okx0jZR6PmFm3ikCCUbNkg)). Attention, CNN is not totally translation invariant as discussed in this paper [Why do deep convolutional networks generalize so poorly to small image transformations?](https://arxiv.org/abs/1805.12177)

## visualization
* t-SNE, T-Distributed Stochastic Neighbouring Entities, [orignal paper](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

* Mask Input and check the predictions. Or local interpretable model-agnostic explanations (LIME)

This repo include several CNN visualization method implemented in Pytorch. [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

* [saliency maps 1312.6034v2](https://arxiv.org/abs/1312.6034v2) The idea is to find pixels need to be changed the least to affect the class score the most. In this paper, they provide two visualizations. One method is to generate an image to maximise the class score and the other method is saliency map. They compute the gradient of output category with respect to input image. All the positive values in the gradients tell us that a small change to that pixel will increase the output value. Hence, visualizing these gradients, which are the same shape as the image should provide some intuition of attention. 

	The rectified/deconv saliency comes from [1311.2901](https://arxiv.org/abs/1311.2901) In deconvnet, "We present a novel way to map these activities back to the input pixel space, showing what input pattern originally caused a given activation in the feature maps." The idea is clipping negative gradients in the backprop step. i.e., only propagate positive gradient information that communicates the increase in output. In guided saliency [1412.6806](https://arxiv.org/abs/1412.6806), the backprop step is modified to only propagate positive gradients for positive activations. Check [keras-vis](https://raghakot.github.io/keras-vis/) for demos with codes. The [./images/guided backprop.png](./images/guided backprop.png) shows the difference very well.

* "Understanding Neural Networks Through Deep Visualization" [1506.06579](https://arxiv.org/abs/1506.06579) two open-source tools are invented. The first one visualizes the activations produced on each layer of a trained convnet. The second one visualizes features at each layer of a DNN via regularized optimization in image space. The regularization methods force to bias images found via optimization toward more visually interpretable examples. Combined several new regularization methods to produce qualitatively clearer, more interpretable visualizations. Tools are in [github](https://github.com/yosinski/deep-visualization-toolbox).

* Deep Dream ([google github](https://github.com/google/deepdream)) The dreaming idea and name became popular on the internet in 2015 thanks to Google's DeepDream program. But the idea has been dates earlier ([wiki](https://en.wikipedia.org/wiki/DeepDream)). Related visualization ideas were developed (prior to Google's work) by several research groups. 

* Visualize higher layer features. Paper "visualizing higher-layer features of a deep network" [2009](https://www.researchgate.net/profile/Aaron_Courville/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network/links/53ff82b00cf24c81027da530.pdf) Old paper related to understand each layer of networks.

* CAM [1512.04150](https://arxiv.org/abs/1512.04150) CNNs trained on classification tasks also have a strong localization ability with the Class Activation Maps (CAM) which are derived from the last feature maps and the weight from FC layers. 
 
* grad-CAM [1610.02391](https://arxiv.org/abs/1610.02391)

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
* Selective search
* Image pyramid
* GrabCut Check the [demo](https://docs.opencv.org/3.1.0/d8/d83/tutorial_py_grabcut.html) in OpenCV.

Before using deep learning as segmentation, people found [TextonForest 2018](http://mi.eng.cam.ac.uk/~cipolla/publications/inproceedings/2008-CVPR-semantic-texton-forests.pdf) and [Random Forest based classifiers 2011](http://www.cse.chalmers.se/edu/year/2011/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf).

### Sigal processing

[Wiener Filter](https://en.wikipedia.org/wiki/Wiener_filter) (维纳滤波) a filter used to produce an estimate of a desired or target random process by linear time-invariant (LTI) filtering of an observed noisy process, assuming known stationary signal and noise spectra, and additive noise.

[Gabor filter](https://en.wikipedia.org/wiki/Gabor_filter) a linear filter used for texture analysis, which means that it basically analyzes whether there are any specific frequency content in the image in specific directions in a localized region around the point or region of analysis. 

[Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) In statistics and control theory, Kalman filtering, also known as linear quadratic estimation (LQE), is an algorithm that uses a series of measurements observed over time, containing statistical noise and other inaccuracies, and produces estimates of unknown variables that tend to be more accurate than those based on a single measurement alone, by estimating a joint probability distribution over the variables for each timeframe. It can be used to fuse data from different sensors to get higher accurate measurements. It is fast, memory friendly. Refer GraphSLAM for most updated methods.

[SLAM](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping) simultaneous localization and mapping (定位与建图). It is the computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it. It is a large topic. 

![](./**.png)


## Chinese demo resources
TO be tested [浅入浅出TensorFlow 8 - 行人分割](https://blog.csdn.net/linolzhang/article/details/70306708) tensorflow with Mask-RCNN