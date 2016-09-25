Copyright (C) 2016 [Sergey Demyanov](http://www.demyanov.net)

contact: my_name@my_sirname.net

You can also find and use my [WorkLab for Tensorflow](https://github.com/sdemyanov/tensorflow-worklab).

This toolbox has been written as a part of my PhD project. It contains the implementation of convolitional neural nets for Matlab, written on C++ and CUDA. The most of the kernels are taken from CUDNN v5 library, others are written manually. Therefore CUDNN, v5 or higher is required.

It also contain the implementation of [Invariant Backpropagation (IBP)](http://arxiv.org/abs/1502.04434) and [Adversarial Training (AT)] (http://arxiv.org/abs/1412.6572) algorithms.


**GENERAL INFORMATION**

Convolutional neural network is a type of deep learning classification and segmentation algorithms, which can learn useful features from raw data by themselves. Learning is performed by tuning its weights. CNNs consist of several layers, which are usually convolutional and subsampling layers following each other. Convolutional layer performs filtering of its input with a small matrix of weights and applies some non-linear function to the result. Subsampling layer does not contain weights and simply reduces the size of its input by averaging of max-pooling operation. The number of channels on the last layer should coincide with the number of classes. If used for classification, the height and width of the last layer output should be 1.

Learning process consists of 2 steps: forward and backward passes, which are conducted for all objects in a training set. On the forward pass each layer transforms the output of the previous layer according to its function. The output of the last layer is compared with the label values and the loss function is computed. On the backward pass the derivatives of loss function with respect to outputs are consecutively computed from the last layer to the first, together with the derivatives with respect to weights. After that the weights are changed in the direction which decreases the value of the loss function. This process is performed for a batch of objects simultaneously, in order to decrease the sample bias. Processing of all objects in the dataset is called the epoch. Usually training consists of many epochs, conducted with different batch splits.

 
**DESCRIPTION**

The toolbox was written for Matlab and its functions can be called only from Matlab scripts. The toolbox requires a Cuda capable GPU. **The toolbox DOES NOT REQUIRE Parallel Computing Toolbox as MatConvNet, but you can import and use pretrained MatConvNet models.** The toolbox operates with 4-dimensional tensors with incides corresponding to height(H), width(W), channel(C) and number(N). Labels should also be 4-dimensional tensors. If used for classification, labels should have height=1 and width=1. Before passing to c++ code the height and width dimensions are permuted, so the layout becomes NCHW (N is the slowest index). Same layout is used for weights everywhere. For speedup purposes weights are passed and returned as a long vector or stretched and concatenated weights from all layers. Use functions **weights = getweights(layers)** and **layers = setweights(layers, weights)** to obtain the vector and assign it back to layers.

The toolbox contains 3 main functions to call:

- **[weights] = genweights(layers, params)**  
Returns randomly generated initial weights for the net. Has to be called before the training.
- **[weights, trainerr] = train(layers, weights, params, train_x, train_y)**  
Performs neural net training. Returns the set of updated weights and values of the main and additional loss functions.
- **[err, bad, pred] = test(layers, weights, params, test_x, test_y)**  
Returns predictions and calculates the test error.


**LAYERS**

Define the structure of CNN. Sets up as cell array, with each element representing an independent layer. Currently 6 layer types are implemented:

- **input** - input layer. Must be the first and only the first one. Must contain the "mapsize" field, that is a vector with 2 integer values, representing the objects size (height and width). May also contain the following additional fields:  
1) 'channels' - that specifies the number of data channels, if it differs from 1.

- **jitt** - jittering layer. Performs affine transformations of the image. With the default parameters performs central cropping. Must have the parameter 'mapsize'. Other possible parameters are:  
1) 'shift' - specifies the maximum shift of the image in each dimension,  
2) 'scale' - specifies the maximum scale in each dimension. Must be more than 1. The image scales with the random factors from [1/x x].  
3) 'mirror' - determines if the image might be mirrored (1) in a particular dimension or not (0).  
4) 'angle' - scalar, that specifies the maximum angle of rotation. Must be from [0, 1]. The value 1 corresponds to 180 degrees.  
5) 'defval' - specifies the value that is used when the transformed image lies outside the borders of the original image. If this value is not specified, the transformed value should be always inside the original one, otherwise there will be an error.
On the test stage the images are just centrally cropped to the size 'mapsize', like there were no additional parameters.

- **conv** - convolutional layer. Must contain the "filtersize" field, that identifies the filter size. Must also contain the "channels" field, which is the number of output channels. If the previous layer has "m" maps and the current one has "n" maps, the total number of filters on it is m * n. Despite that it is called convolutional, it performs filtering, that is a convolution operation with flipped dimensions.

- **deconv** - reverse convolutional layer. Must contain the same fields as the convolutional layer. On the forward pass performs the same operation as performed on the backward pass of the "conv" layer, and otherwise. Therefore, instead of scaling the dimensions by a factor of "stride" it multiplies them on "stride".

- **pool** - pooling layer. The pooling type is specified by "pooling" field, which can be eigther "max" or "avg". Default value is "max". Must contain the "scale" and "stride" fields, which are the vectors with 2 integer values.

- **full** - fully connected layer. Produces a tensor with height=1 and width=1. Must contain the "channels" field, which defines the number of output channels. Considers its input as a single vector.


Additionally, all layers might have the following parameters:

- **function** - defines the non-linear transformation function. It can be "relu", "sigm", "soft" or "none", which correspond to rectified linear unit, sigmoid, softmax or no transformation respectively. The default value is "relu". The value "soft" must be used only on the last layer.

- **padding** - a 2-dimensional vector of non-negative integers. Considered by "conv", "deconv" and "pool" layers. Determines the number of zero padding rows (columns) on the top and bottom (padding[0]) and left and right (padding[1]).

- **stride** - a 2-dimensional vector of non-negative integers. Considered by "conv", "deconv" and "pool" layers. Determines the distance between the positions of applied kernels in vertical and horizontal dimensions.

- **init_std** - the standard deviation of normal distribution that is used to generate the weights. When is not defined, the init_std = $\sqrt{2/n_{in}}, n_in = h * w * m$, where 'h' and 'w' is the filter size and 'm' is the number of input channels. Considered by all layers with weights.

- **add_bias** - whether the layer should add bias to the output or not. The length of the bias vector is equal to the number of output channels. Considered by all layers. Default is true for all layers with weights, false for others.

- **bias_coef** - the multiplier for the bias learning rate. Default is 1.

- **lr_coef** - the multiplier for the learning rate on this layer, both weights and biases. Considered by all layers. Set it to 0 to fix some layers.

- **dropout** - a scalar from [0, 1), which determines the probability of dropping the activations on this layer. Should not be too large, otherwise it drops everything.


**PARAMS**

Define the learning process. It is a structure with the fields described below.

- **seed** - any integer, which allows to repeat the same random numbers. Default is 0. Note that if "conv", "deconv" or "pool" layers are used, the results are not guaranteed to be exactly the same even if the same seed is used. For more details read CUDNN User Guide.

- **batchsize** - defines the size of batches. Default is 32.

- **epochs** - the number of repeats the training procedure with different batch splits. Default is 1.

- **alpha** - defines the learning rate. Default is 1. 

- **beta** - defines the invariant learning rate (see the [article](http://arxiv.org/abs/1502.04434)). The value '0' corresponds to the standard backpropagation algorithm. Default is 0. 

- **shift** - defines the shift in the Adversarial Training algorithm (see the [article](http://arxiv.org/abs/1412.6572)). The value '0' corresponds to the standard backpropagation algorithm. Default is 0. 

- **normfun** - defines the type of norm used as second loss function in IBP or used to generate adversarial examples in AT. Default is 1.

- **momentum** - defines the actual direction of weight change according to the formula m ** dp + (1-m) ** d, where m is momentum, dp is the previous change and d is the current derivative. Default is 0. 

- **decay** - defines the weight decay, i.e. every update all weights are multiplied on (1-decay).

- **lossfun** - string. Specifies the employed loss function. Must be eigher "squared" or "logreg", that correspond to sum of squared differences and negative log likelihood respectively. If you use "logreg", it is better to use "softmax" nonlinear function on the last layer and reduce the learning rate about 10 times. The default value is "logreg".

- **shuffle** - determines whether the input dataset will be shuffled or not. If it is set to 0, the batches are created in a natural order: first "batchsize" objects become the first batch and so on. Otherwise, it should be 1. Default is 0.

- **verbose** - determines output info during learning. For 0 there is no output, for 1 it prints only number of current epoch, for 2 it prints both numbers of epoch and batch. Default is 0.

- **memory** - determines the maximum number of megabytes of GPU memory allocated as a workspace for convolutional operations. Default is 512.

- **gpu** - allows to specify the index of gpu device to work on. Default is 0.

- **classcoefs** - allow to specify coefficients of class importance, for example if the dataset is unbalanced. Should be a vector of 1xN, where N is the number of classes. By default all coefficients are 1. Recommended class coefficients for an unbalanced dataset are $c_i = (\sum_i^N n_i / n_i)/N$.


**COMPILATION**

If you cannot use the provided binaries, you need to compile them by yourself. The compilation options are defined in the file "settings.h". They are:

- **PRECISION**. Might have two values: 1 - single, uses type 'float'. 2 - double, uses type 'double'. The second version has not been tested.

- **PRECISION_EPS**. Equal to 1e-6 by default. For consistency purposes all values that are less than it are assigned to 0.


**COMPILATION**

- **Linux** - adjust the paths in the './c++/Makefile' file and run "make". That should be enough.  

- **Windows** - has been tested long time ago.

1) Using 'compile' script. 
While CPU compilation is easy, the GPU compilation is tricky and might take some efforts to do it.
First of all, run 'mex -setup' in order to check that you have a proper C++ compiler. If not, install it. You need either a full version of Visual Studio or an express version with Microsoft SDK, that are free. Of course, you need to install CUDA as well. Download it from NVIDIA site. The CUDA settings for 'mex' are located in file with the name like "mex_CUDA_win64.xml". Read more on the MathWorks [website](http://www.mathworks.com.au/help/distcomp/run-mex-functions-containing-cuda-code.html#btrgjh3-1). You must have this file in your Matlab folder. The one that works for me is located in "./c++/cuda" folder. Adjust your Microsoft SDK and CUDA folders, CUDA computation capability and other options there. Make sure you have proper values of environment variables 'CUDA_PATH' and 'VS100COMNTOOLS'. You can do it using functions 'getenv' and 'setenv'. If you don't do it, you might get an error "No supported compiler or SDK was found". You might also get an error about the file 'vcvars64.bat'. In this case use the one that is located in "./c++/cuda" folder. Adjust the path in it as well. After that you should be able to compile.

2) Using Visual Studio project.  
This is a project to compile 'cnntrain_mex'. Add all '.h', '.cpp' and '.cu' files, adjust paths in Include and Libraries fields, and enjoy incremental compilation every time you change just one single file. Create similar project with the same settings to compile 'classify' and 'genweights'.


**LOADING PRETRAINED WEIGHTS**

It is possible to use pretrained models from [MatConvNet](http://www.vlfeat.org/matconvnet/). Given that you reconstruct the same architecture, you can use the function 'import_weights.m' to load the pretrained weights to the network. An example for fully convolutional network is provided.


**EXAMPLES**

- **mnist.m** - provides an example of training a convolutional network on MNIST dataset. The error after 5 epochs should be close to 1%.

- **fcn_test.m** - provides an example of loading [pretrained weights](http://www.vlfeat.org/matconvnet/models/pascal-fcn32s-dag.mat) from MatConvNet and segmenting the images. On the provided test set, which is smaller than the original PASCAL test set, the results should be (meanIU = 0.5188, pixelAccuracy = 0.8766, meanAccuracy = 0.6574). This is because one of the classes is not presented, so its IU is 0.


**KNOWN ERRORS**

- When you change gpu index, the first time it might fail. Just run it again.

