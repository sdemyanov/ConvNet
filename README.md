Copyright (C) 2014 Sergey Demyanov

contact: sergey@demyanov.net

http://www.demyanov.net

!!! UPDATE !!! The [Invariant Backpropagation](http://arxiv.org/abs/1502.04434) (IBP) algorithm  is implemented.

!!! IMPORTANT !!! The GPU version works only on devices with the compute capability >= 3.0

This library has been written as a part of my PhD project. It contains the implementation of convolitional neural nets for Matlab, written on Matlab, C++ and CUDA for CPU and GPU processing. All versions work identically. The GPU version uses kernels from Alex Krizhevsky's library [cuda-convnet2](https://code.google.com/p/cuda-convnet2/), so it is _really_ fast. In some cases it is about 400 times faster than CPU version.

It also contain the implementation of the [Invariant Backpropagation](http://arxiv.org/abs/1502.04434) (IBP) algorithm , that allows to train transformation-invariant neural networks. The algorithm is described in the article "Invariant backpropagation: how to train a transformation-invariant neural network".

GENERAL INFORMATION

Convolutional neural net is a type of deep learning classification algorithms, that can learn useful features from raw data by themselves. Learning is performed by tuning its weighs. CNNs consist of several layers, that are usually convolutional and subsampling layers following each other. Convolution layer performs filtering of its input with a small matrix of weights and applies some non-linear function to the result. Subsampling layer does not contain weights and simply reduces the size of its input by averaging of max-pooling operation. The last layer is fully connected by weights with all outputs of the previous layer. The output is also modified by a non-linear function. If your neural net consists of only fully connected layers, you get a classic neural net.

Learning process consists of 2 steps: forward and backward passes, that repeat for all objects in a training set. On the forward pass each layer transforms the output from the previous layer according to its function. The output of the last layer is compared with the label values and the total error is computed. On the backward pass the corresponding transformation happens with the derivatives of error with respect to outputs and weights of this layer. After the backward pass finished, the weights are changed in the direction that decreases the total error. This process is performed for a batch of objects simultaneously, in order to decrease the sample bias. After all the object have been processed, the process might repeat for different batch splits.

 
DESCRIPTION

The library was written for Matlab and its functions can be called only from Matlab scripts. It operates with 2(3)-dimensional objects, like images, that are stored as a 3(4)-dimensional array. The last index represents the object index. The labels must be in a 2-dimensional array where the second index represent the class label (0 or 1). Normally there is only one "1" in a row.

The library contains 3 main functions to call:

- [weights] = genweights(layers, seed, funtype);
Returns randomly generated initial weights for the net. Has to be called before the training.
- [weights, trainerr] = cnntrain(layers, weights, params, train_x, train_y, type);  
Performs neural net training. Returns weights from all layers as a single vector.
- [err, bad, pred] = cnntest(layers, weights, params, test_x, test_y, type)
Calculates the test error. Based on cnnclassify, that returns only the predictions.


Parameters:

layers - the structure of CNN. Sets up as cell array, with each element representing an independent layer. Layers can be one of 5 types:

- i - input layer. Must be the first and only first one. Must contain the "mapsize" field, that is a vector with 2 integer values, representing the objects size. May also contain the following additional fields:  
1) 'outputmaps' - that specifies the number of data channels, if it differs from 1.  
2) 'norm' - determines the desired norm of input vectors. It performs a normalization within a sample.  
3) 'mean' - determines the desired mean value of each feature across all samples.  
4) 'maxdev' - determines the maximum standard deviation of each feature across all samples.

- j - jitter layer. Specify possible transformations of the input maps, that might be used to achieve transformation invariance. Must have the parameter 'mapsize'. Other possible parameters are:  
1) 'shift' - specifies the maximum shift of the image in each dimension,  
2) 'scale' - specifies the maximum scale in each dimension. Must be more than 1. The image scales with the random factors from [1/x x].  
3) 'mirror' - binary vector, that determines if the image might be mirrored in a particular dimension or not.  
4) 'angle' - scalar, that specifies the maximum angle of rotation. Must be from [0, 1]. The value 1 corresponds to 180 degrees.  
5) 'defval' - specifies the value that is used when the transformed image lies outside the borders of the original image. If this value is not specified, the transformed value should be always inside the original one, otherwise there will be an error. This layer is not implemented in Matlab version.  
On the test set the images are just centrally cropped to the size 'mapsize', like there were no additional parameters.

- c - convolutional layer. Must contain the "filtersize" field, that identifies the filter size. Must not be greater than the size of maps on the previous layer. Must also contain the "outputmaps" field, that is the number of maps for each objects on this layer. If the previous layer has "m" maps and the current one has "n" maps, the total number of filters on it is m * n. Despite that it is called convolutional, it performs filtering, that is a convolution operation with flipped dimensions. May contain the following additional fields:  
1) 'padding' - specifies the size of zero padding around the maps for each dimension. The default value is 0.  
2) 'initstd' - the standard deviation of normal distribution that is used to generate the weights. The default value is 0.01. Biases are always initialized by 0.  
3) 'biascoef' - specifies the multiplier for bias learning rate. Might be used if for some reason you decided to use another learning rate than for other weights. The default value is 1.

- s - scaling layer. Reduces the map size by pooling. Must contain the "scale" field, that is also a vector with 2 integer values. May additionally contain 'stride' field, that determines the distance between neighbouring blocks in each dimension. By default is equal to 'scale'.

- f - fully connected layer. Must contain the "length" field that defines the number of its outputs. The last layer must have this type. For the last layer the length must coincide with the number of classes. May also contain the following additional fields:  
1) "dropout" - determines the probability of dropping the activations on this layer. Cannot be used on the last layer. Should not be too large, otherwise it drops everything.  
2) 'initstd' - the same as for convolutional layers. The default value is 0.1.  
3) 'biascoef' - the same as for convolutional layers.

All layers except "i" may contain the "function" field, that defines their action. For:
- c and f - it defines the non-linear transformation function. It can be "soft", "sigm" or "relu", that correspond to softmax, sigmoid and rectified linear unit respectively. The default value is "relu". The value "soft" must be used only on the last layer.
- s - it defines the pooling procedure, that can be either "mean" or "max". The default value is "mean". 

params - define the learning process. It is a structure with the following fields. If some of them are absent, the value by default is taken.
- seed - any non-negative integer, that allows to repeat the same random numbers. Default is 0.
- batchsize - defines the size of batches. Default is 128.
- epochs - the number of repeats the training procedure with different batch splits. Default is 1.
- alpha - defines the learning rate. Default is 1. Can also be the vector of the length 'epochs'. Then an individual rate for each epoch is used.
- beta - defines the invariant learning rate (see the [article](http://arxiv.org/abs/1502.04434)). The value '0' corresponds to the standard backpropagation algorithm. Default is 0. Can also be the vector of the length 'epochs'. Then an individual rate for each epoch is used.
- momentum - defines the actual direction of weight change according to the formula m * dp + (1-m) * d, where m is momentum, dp is the previous change and d is the current derivative. Default is 0. Can also be the vector of the length 'epochs'. Then an individual momentum for each epoch is used.
- adjustrate - defines how much we change the learning rate for a particular weight. If the signs of previous and current updates coincide we add it to the learning rate. If not, we divide the learning rate on (1 - adjustrate). Default is 0.
- maxcoef - defines the maximum and minimum learning rates, that are alpha * maxcoef and alpha / maxcoef respectively. Default is 1.
- balance - boolean variable. Balances errors according to the class appearance frequencies. Useful for highly unbalanced datasets. Default is 0.
- lossfun - string. Specifies the employed loss function. Must be eigher "squared" or "logreg", that correspond to sum of squared differences and negative log likelihood respectively. If you use "logreg", it is better to use "softmax" nonlinear function on the last layer and reduce the learning rate about 10 times. The default value is "squared".
- shuffle - determines whether the input dataset will be shuffled or not. If it is set to 0, the batches are created in a natural order: first "batchsize" objects become the first batch and so on. Otherwise, it should be 1. Default is 0.
- verbose - determines output info during learning. For 0 there is no output, for 1 it prints only number of current epoch, for 2 it prints both numbers of epoch and batch. Default is 0.

weights - the weights vector obtained from genweights or cnntrain, that is used for weights initialization. Can be used for testing or continuing the training procedure. 

funtype - defines the actual function that is used. Can be either "gpu", "cpu" or "matlab". While "matlab" is slower, it is easier to do some debugging and see the intermediate results.

COMPILATION

If you cannot use the binaries for C++ CPU and GPU versions, you need to compile them by yourself. The compilation options are defined in the file "settings.h". Here they are:

- COMP_REGIME. Identifies the version you want to compile. Might have the following values:  
1) 0 - compiles the single-thread CPU version. Use it if you don't have GPU with CUDA support :)  
2) 1 - compiles the multi-thread GPU version. Use it to speed-up computations. However, if you run several Matlabs in parallel, I would recommend to use the single-thread version.  
3) 2 - compiles the GPU version. The main one.

- PRECISION. Might have two values: 1 - single, uses type 'float'. 2 - double, uses type 'double'. The GPU version supports only single precision.

- PRECISION_EPS. Equal to 1e-6 by default. For consistency purposes all values that are less than it are assigned to 0. In Matlab it is defined in cnnsetup.m.

There are two ways to compile "mex" files. First of them is to run the 'compile' script in the main folder. Run in with the single parameter that specifies the regime. IMPORTANT! Make sure you have the same value in the "settings.h". If they do not correspond, you might get either compilation errors or the GPU version that is performed on CPU. You may also specify the second parameter that identifies the particular files. Use 1 for 'cnntrain', 2 for 'classify', 3 for 'genweights'. While the compilation process takes just several seconds for CPU version, it takes several minutes to compile the GPU version. Please, be patient. If there is an error, you will see red messages.

Alternatively, if you need just a GPU version, you can use the Makefile in Linux, or try to use my project for Visual Studio 2012 in Windows.


COMPILATION FOR WINDOWS

- Using 'compile' script.  
While CPU compilation is easy, the GPU compilation is tricky and might take some efforts to do it.
First of all, run 'mex -setup' in order to check that you have a proper C++ compiler. If not, install it. You need either a full version of Visual Studio or an express version with Microsoft SDK, that are free. Of course, you need to install CUDA as well. Download it from NVIDIA site. The CUDA settings for 'mex' are located in file with the name like "mex_CUDA_win64.xml". Read more on the mathworks [website](http://www.mathworks.com.au/help/distcomp/run-mex-functions-containing-cuda-code.html#btrgjh3-1). You must have this file in your Matlab folder. The one that works for me is located in "./c++/cuda" folder. Adjust your Microsoft SDK and CUDA folders, CUDA computation capability and other options there. Make sure you have proper values of environment variables 'CUDA_PATH' and 'VS100COMNTOOLS'. You can do it using functions 'getenv' and 'setenv'. If you don't do it, you might get an error "No supported compiler or SDK was found". You might also get an error about the file 'vcvars64.bat'. In this case use the one that is located in "./c++/cuda" folder. Adjust the path in it as well. After that you should be able to compile.

- Using Visual Studio project.  
This is a project to compile 'cnntrain_mex'. Add all '.h', '.cpp' and '.cu' files, adjust paths in Include and Libraries fields, and enjoy incremental compilation every time you change just one single file. Create similar project with the same settings to compile 'classify' and 'genweights'.


COMPILATION FOR LINUX

- Using 'compile' script.  
If you want to compile via "compile" script, which uses "mex" function, first you need to specify your compiler. This how you do it:  
1) Install the version of "gcc" and "g++", supported by your version of Matlab,  
2) Copy the settings file "MatlabRoot"/bin/mexopts.sh to ~/.matlab/"MatlabVersion",  
3) In this file change the values "CC" and "CXX" to these versions, like "CXX='g++-4.7'.  
4) Find and change the line "CXXFLAGS='-ansi -D_GNU_SOURCE'" to "CXXFLAGS='-std=c++11 -D_GNU_SOURCE'". The code uses c++11 features, so you need to make "g++" to understand them.

- Using Makefile.  
In order to compile the GPU version, adjust the paths in the './c++/Makefile' file and run "make". That should be enough.  

If you have problems with loading '.so' files, make sure you have CUDA library folder (usually '/usr/local/cuda/lib64') in the variable LD_LIBRARY_PATH. You can check it by the Matlab 'getevn' command.


NOTICE

Random numbers are used for generating weights, shuffling batches and dropout. Thus, if want to get identical resutls in all Matlab and C++ versions, you need to use the same initial weights, do not use shuffling and set dropout to 0. Note, that these versions produce different random numbers for the same seeds.

ACKNOLEDGEMENTS

- Almost all CUDA kernels were taken from the library of Alex Krizhevsky [cuda-convnet2](https://code.google.com/p/cuda-convnet2/). Enormous thanks to Alex for saving so much time.

- The C++ version was inspired by [Yichuan Tang](http://www.cs.toronto.edu/~tang) and his [solution](http://code.google.com/p/deep-learning-faces/) for the Kaggle Facial Expression Recognition Challenge. The structure of C++ code was originated from there.

- The original Matlab code was created by [Rasmus Berg Palm](dtu.academia.edu/RasmusBergPalm) and can be found in his [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox). The Matlab version basically remained the same structure as there.
 
