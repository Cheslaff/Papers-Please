## Notes:

Problem of computer vision consists of 2 parts:
1) Feature Extractor
2) Classifier
It's potentially interesting to rely on Fully-Connected
networks in feature extractor, but there are major drawbacks:
1) Images are large => A lot of parameters used
To learn this many weights we need to increase the size of our dataset.
2) Topology of the input is entirely ignored.

FC nets ignore local correlations in our data.
**Solution: Convolutional Networks**

Approach:
- using local receptive fields
- using shared weights
- using temporal sub-sampling


We extract feature maps from the input image applying convolutional kernels to it. Kernels are later learned to extract meaningful features.
Interesting property of convolution is "shift-robustnes". If object on the image shifts, its representation also shifts.
However, precise position is harmful, because on different samples of the data corners and edges are gonna be located in different places.
To avoid this position precision we apply *subsampling* operation that reduces the size of each feature map (feature map - output channel)

LeNet-5:
```
The second hidden layer of LeNet
5 is a subsampling layer This layer comprises six feature
maps, one for each feature map in the previous layer The
receptive field of each unit is a 2 by 2 area in the previous
layer's corresponding feature map Each unit computes the
average of its four inputs, multiplies it by a trainable coef
ficient, adds a trainable bias, and passes the result through
a sigmoid function. Contiguous units have nonoverlapping
contiguous receptive fields
```
Didn't know about that🔝.
Probably it aged like milk and died off.

Since all the weights are learned with backpropagation,
convolutional networks can be seen as **synthesizing their
own feature extractor**


<br>
<img src="https://production-media.paperswithcode.com/methods/LeNet_Original_Image_48T74Lc.jpg"><br>
7 Layers.
Instead of Flatten it uses convolution operation with 5x5 kernel and 120 feature maps that flattens representation.
120 units "flattened" layer is followed by 84 units hidden layer.
Layers are activated with tanh except the last layer with 10 units.
It uses RBF-like activation:

$$y_i = \sum_j{(x_j - w_{ij})}^2$$ 