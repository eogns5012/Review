Review : Focal Loss for Dense Object Detection
==============================================

Introduction
------------

### Two-stage object detectors

-	Current state-of-the-alt object detectors (proposal-driven mechanism)

-	R-CNN, FPN(Feature Pyramid Network)

-	**First stage**

	-	generate set of candidate locations (region proposal)
	-	Selective search, EdgeBoxes, DeepMask, RPN

-	**Second stage**

	-	classifies each candidate locations as one of the foreground classes or background using CNN

-	Address class imbalance

	-	First stage : reduce the number of candidate locations filtering out most background samples
	-	Second stage : maintain balance between foreground and background
	-	fixed foreground-to-background ratio
	-	online hard example mining(OHEM)

### One-stage object detectors

-	YOLO, SSD

-	Faster but lower accuracy than two-stage object detectors

-	**extreme foreground-background class imbalance**

	-	larger set of candidate object locations regularly sampled across an inage

### New loss function : Focal loss

![Figure1](https://user-images.githubusercontent.com/56924420/135286640-375c8b11-41f3-4804-8890-4461e93f7dff.PNG)

-	Designed to address extreme foreground and background class imbalance in one-stage object detectors

-	Scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases

	-	Automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples

### RetinaNet

-	Simple one-stage object detector to demonstrate the effectiveness of the focal Loss

-	Based on ResNet-101-FPN backbone and use anchor boxes

---

Related Work
------------

### Classic Object Detectors

-	Based sliding-window

-	HOG(Histogram of gradient), DPM(Deformable part model)

### Two-stage detectors

-	Selective search

	-	First stage generate a sparse seet of candidate proposals that should contain all objects while filtering out the majority of negative locations
	-	Second stage classifies the proposals into foreground classes / background

-	R-CNN

	-	Upgrade the classifier to a convolutional network

-	Faster R-CNN

	-	Use RPN(Region Proposal Networks)

### One-stage detectors

-	SSD (Single Shot multibox Detector)

	-	10 ~ 20% lower AP than two-stage methods

-	YOLO (You Only Look Once)

	-	Focus on an even more extreme speed/accuracy trade-off

### RetinaNet

-	Shares many similarites with previous detectors

	-	'Anchors' introduced RPN
	-	Feature pyramid in SSD and FPN

-	**Its achieves top results not based on innovations in network design but due to focal loss**

### Class imbalance

-	Classic(DPMs) and recent(SSD) methods face a large class imbalance during training

-	Evaluate 10k~100k candidate locations per image but only a few locations contain objects

-	Two problems

	1.	Training is inefficient as most locations are easy negative that contribute no useful learning signal
	2.	Easy negative can overwhelm training and lead to degenerate models

-	A common solution is hard negative mining

	-	Samples hard examples during training

-	In contrast, focal loss naturally handles the class imbalance without sampling and without easy negatives overwhelming the loss and computed gradients

### Robust estimation

-	Designing robust loss functions(Huber loss) that reduce the contribution of outliers by down-weighting the loss of examples with large errors(hard examples)

-	In contrast, focal loss that reduce the contribution of inliers by down-weighting the loss of examples with small errors(easy examples)

-	Focal loss and robust loss have opposite roles

	-	Focal loss focuses training on a sparse set of hard examples

---

Focal Loss
----------

### Binary Cross Entropy

$$ \text{CE}(p,y) = \begin{cases} -\text{log}(p) & \text{if} \; y = 1 \\ -\text{log}(1-p) & \text{otherwise} \end{cases} $$

-	$y \in \{ \pm 1\}$ specifies the ground-truth class

-	$p \in [0,1]$ is the model's estimated probability for the class with label $y=1$

$$ p_t = \begin{cases} p & \text{if} \; y = 1 \\ 1-p & \text{otherwise} \end{cases} $$

-	Hence $\text{CE}(p,y) = \text{CE}(p_t) = - \text{log}(p_t)$

-	One notable property of CE is that even examples that are easily classified ($p_t \gg 0.5$) incur a loss with non-trivial magnitude

-	When summed over a large number of easy examples, these small loss values can overwhelm the rare class

### Balanced Cross Entropy

$$ \text{CE}(p_t) = - \alpha_t \text{log}(p_t)$$

-	$\alpha$ is a weighting factor
	-	$\alpha \in [0,1]$ for class 1, and $1 - \alpha$ for class -1

![BCE_loss_plot](https://user-images.githubusercontent.com/56924420/135324067-cb681504-8e92-4140-b10b-da0cb3aaec6d.png)

### Focal Loss Definition

-	While $\alpha$ balances the importance of positive/negative examples, it does not differentiate between easy/hard examples

-	Reshape the loss function to down-weight easy examples and thus focus training on hard negatives

$$ \text{FL}(p_t) = -(1- p_t)^\gamma \text{log}(p_t)$$

-	$(1-p_t)^\gamma$ is a modulating factor, $\gamma \ge 0$ is tuneable parameter

-	Two properties of the focal loss

	1.	When an example is misclassified and $p_t$ is small, the modulating factor is near 1 and loss is unaffected $p_t \rightarrow 1$, factor goes to 0 and the loss for well-classified exampls is down-weighted

	2.	When $\gamma$ is increased, the effect of the modulation factor is increased

-	The modulating factor reduces the loss contribution from easy examples

$$ \text{FL}(p_t) = -{\alpha_t}(1- p_t)^\gamma \text{log}(p_t)$$

-	$\alpha$-balanced variant of the focal loss

### Class imbalance and Model Initialization

-	Binary classification models are by default initialized to have equal probability of outputting either $y = -1 \text{or} \, 1$

-	We introduce the concept of a 'prior' for the value of $p$ estimated by the model for the rare class at the start or training

-	Denote the prior by $\pi$

-	This improve training stability for both the cross entropy and focal loss in the case of heavy class imbalance

---

RetinaNet Detector
------------------

![Figure3](https://user-images.githubusercontent.com/56924420/135328928-ebf3ed2b-54af-4054-99be-ad193461c931.PNG)

-	RetinaNet is a single, unified network composed of a **backbone network** and **two task-specific subnetworks**

### Feature Pyramid Network Backbone

-	FPN create multi-scale feature pyramid from a single resolution input image

-	Each level of the pyramid can be used for detecting objects at a different scale

-	FPN improves multi-scale predictions from fully convolutional networks

-	Build FPN on top of the ResNet architecture

-	pyramid with levels $P_3$ through $P_7$, all pyramid levels have $C = 256$ channels

### Anchors

-	Each pyramid level use anchors at three aspect ratio {$1:2, 1:1, 2:1$}
-	Each pyramid level add anchors of sizes {$2^0, 2^{1/3}, 2^{2/3}$} of the original set of 3 aspect ratio anchors
-	Total $A = 9$ anchors per level and across levels they cover the scale range 32-813 pixels
-	Anchors are assigned to ground-truth object boxes using an IoU threshold of 0.5 and to background if their IoU is in [0, 0.4)
-	If IoU is in [0.4, 0.5), it is ignored during training

### Classification Subnet

-	Predicts the probability of object presence at each spatial position for each of the $A$ anchors and $K$ object classes
-	This subnet is a small FCN attached to each FPN level
-	Applies four $3 \times 3$ conv layers, each with $C$ filters and each followed ReLU activation

### Box Regression Subnet

-	In parallel with the object classification subnet, we attach another small FCN to each pyramid level for the purpose of regressing the offset from each anchor box to a nearby ground-truth object
-	4 offset are (x_center, y_center, width, height)
-	Use class-agnostic bounding box regressor
	-	Uses fewer parameter and effective

Inference and Training
----------------------

### Inference

-	To improve speed, we only decode box predictions from at most 1k top-scoreing predictions per FPN level, after thresholding detector confidence at 0.05
-	Final detections, use non-maximum suppression with a threshold of 0.5

### Focal Loss

-	Use the loss on the output of the classification subnet
-	Focal loss is applied to all ~100k anchors in each sampled image and is computed as the sum of all anchors
-	Note that $\alpha$, the weight assigned to the rare class, also has a stable range, but it interacts with $\gamma$ making it necessary to select the two together

-	$\gamma = 2,\, \alpha = 0.25$ work best

### Initialization

-	Experiment with ResNet-50-FPN and ResNet-101-FPN backbone
-	Models are pre-trained on ImageNet1K
-	All new conv layers except the final one in the RetinaNet subnets are initialized with bias $b = 0$ and a Gaussian weight fill with $\sigma = 0.01$
-	For the final conv layer of the classification subnet, set the bias Initialization to $ b = -\text{log}((1- \pi)/\pi),\, \pi = 0.01$

### Optimization

-	Trained with stochastic gradient descent(SGD) over 8 GPUs with a total of 16 images per minibatch
-	Initial learning rate = 0.01, total 90k epochs
-	at 60k and 80k, learning rate = 0.001, 0.0001 (divided by 10)
-	weight decay = 0.001, momentum = 0.9
-	class predict : focal loss / box regression : standard smooth $L_1$ loss
-	Training time ranges between 10 and 35 hours

---

Experiments
-----------

Training Dense Detection
------------------------

### Network Initialization

-	First uses standart cross entropy(CE) loss but fails quickly with the network diverging during training
-	Simply initializing the last layer of our model such that the prior probability of detecing an object is $\pi = 0.01$ enables effective learning
	-	AP 30.2 on COCO
-	so we use $\pi = 0.01$ for all experiments

### Balanced Cross Entropy

-	Next uses $\alpha$-balanced CE loss

![Table1-(a)](https://user-images.githubusercontent.com/56924420/135340066-ffd074ce-463b-49fe-ae51-62d60f8cd220.PNG)

-	$\alpha = 0.75$ gives of 0.9 points AP

### Focal Loss

![Table1-(b)](https://user-images.githubusercontent.com/56924420/135341584-fe4c3d64-e775-4772-bb3e-53c99d5b02ac.PNG)

-	For a fair comparison we find the best $\alpha$ for each $\gamma$
-	The benefit of chaninging $\gamma$ is much larger, and indeed the best $\alpha's$ ranged in just [0.25, 0.75]
-	We use $\gamma= 2.0 \, \text{with} \, \alpha = 0.25$, but $\alpha = 0.4$ works nearly as well

### Analysis of the Focal Loss

-	To understand the focal loss better, we analyze the emprical dirtribution of the loss of a converged model

-	We take our default ResNet-101 600-pixel model trained with $\gamma = 2$

![Figure4](https://user-images.githubusercontent.com/56924420/135345787-53026669-7680-4eec-920a-9f36dac149a3.PNG)

-	Cumulative distribution functions for positive and negative samples
-	Observe the positive samples, we see that the CDF looks fairly similar for different values of $\gamma$
-	The effect of $\gamma$ on negative samples is dramatically different
-	FL can effectively discount the effect of easy negatives, focusing all attention on the hard negative examples

### Online Hard Example Mining(OHEM)

-	Like the focal loss, OHEM puts more emphasis on misclassified example, but completely discards easy examples

![Table1-(d)](https://user-images.githubusercontent.com/56924420/135347622-13fec800-c0fa-4183-ba50-94caa3f0bb93.PNG)

Model Architecture Design
-------------------------

### Anchor Density

-	One-stage detectors use a fixed sampling grid, use multiple 'anchors' at each spatial position to cover boxes of various scales and aspect ratios

![Table1-(c)](https://user-images.githubusercontent.com/56924420/135349056-15a10520-b2b4-4d72-b93c-43e6c810723f.PNG)

### Speed versus Accuracy

![Figure2](https://user-images.githubusercontent.com/56924420/135349250-97ca9ec5-3284-4ee1-8061-11930d111ab5.PNG)

![Table1-(e)](https://user-images.githubusercontent.com/56924420/135349180-b037bbb8-14c6-4d30-bac6-757474776cd4.PNG)

### Comparison to State of the Art

![Table2](https://user-images.githubusercontent.com/56924420/135349678-80dbd607-ea73-43eb-867d-db5a982e0643.PNG)

---

Conclusion
----------

-	We identify class imbalance as the primary obstacle preventing one-stage object detectors from surpassing top-perfoming, two-stage methods
-	Focal loss which applies a modulating term to the croee entropy loss in order to focus learning on hard negative examples
