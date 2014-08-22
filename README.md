Copyright (C) 2014 Sergey Demyanov

contact: sergey@demyanov.net

http://www.demyanov.net

This library has been written as a part of my project on facial expression analysis. It contains the implementation of convolitional neural nets for Matlab, both on Matlab and C++. The C++ version works about 2 times faster. Both implementations work identically.

GENERAL INFORMATION

Convolutional neural net is a type of deep learning classification algorithms, that can learn useful features from raw data by themselves. Learning is performed by tuning its weighs. CNNs consist of several layers, that are usually convolutional and subsampling layers following each other. Convolution layer performs filtering of its input with a small matrix of weights and applies some non-linear function to the result. Subsampling layer does not contain weights and simply reduces the size of its input by averaging of max-pooling operation. The last layer is fully connected by weights with all outputs of the previous layer. The output is also modified by a non-linear function. If your neural net consists of only fully connected layers, you get a classic neural net.

Learning process consists of 2 steps: forward and backward passes, that repeat for all objects in a training set. On the forward pass each layer transforms the output from the previous layer according to its function. The output of the last layer is compared with the label values and the total error is computed. On the backward pass the corresponding transformation happens with the derivatives of error with respect to outputs and weights of this layer. After the backward pass finished, the weights are changed in the direction that decreases the total error. This process is performed for a batch of objects simultaneously, in order to decrease the sample bias. After all the object have been processed, the process might repeat for different batch splits.

 
DESCRIPTION

The library was written for Matlab and its functions can be called only from Matlab scripts. It operates with 2-dimensional objects, like images, that are stored as a 3-dimensional array. The last index represents the object number. The labels must be in a 2-dimensional array where the first index represents the class label (0 or 1) for each object.

The library contains 3 main functions to call:

- [weights] = genweights(layers, funtype);
Returns randomly generated initial weights for the net. Has to be called for the before the training.
- [weights, trainerr] = cnntrain(layers, weights, train_x, train_y, params, funtype);  
Performs neural net training. Returns weights from all layers as a single vector.
- [err, bad, pred] = cnntest(layers, weights, test_x, test_y, funtype)
Calculates the test error. Based on cnnclassify, that returns only the predictions.


Parameters:

layers - the structure of CNN. Sets up as cell array, with each element representing an independent layer. Layers can be one of 4 types:
- i - input layer. Must be the first and only first. Must contain the "mapsize" field, that is a vector with 2 integer values, representing the objects size. May also contain the "outputmaps" field, that specifies the number of data channels. The 'norm' field corresponds to the desired norm of input vectors. It performs a normalization within a sample. May also contain 'mean' and 'maxdev' fields, that indicate the desired mean value and maximum standard deviation of each feature across all samples. See Matlab 'initnorm' function for better understanding.
- j - jitter layer. Specify possible transformations of the input maps, that might be used to avoid transformation invariance. The possible parameters are: 'shift' field, that specify the allowed shift of the image in each dimension, 'scale' - scale in each dimension. If for some dimension it is x>1, the image might be scaled with the factor from [1/x x]. The 'mirror' vector is binary and determines if the image might be mirrored in a particular dimension or not. The 'angle' parameter specifies allowed angle of rotation. Pi corresponds to the value 1. Layer must contain the 'mapsize' field, that is typically smaller than the input map size. If the transformed image might be out of the original one, there will be an error. If you want to avoid it, specify the 'default' parameter, that defines the maps value outside the borders. This layer is fully implemented only in C++ vesion, in Matlab it can only crop the image to the mapsize.
- c - convolutional layer. Must contain the "kernelsize" field, that identifies the filter size. Must not be greater than the size of maps on the previous layer. Must also contain the "outputmaps" field, that is the number of maps for each objects on this layer. If the previous layer has "m" maps and the current one has "n" maps, the total number of filters on it is m * n. Despite that it is called convolutional, it performs filtering, that is a convolution operation with flipped dimensions. May contain 'padding' field, that specifies the size of zero padding around the maps for each dimension. The default value is 0.
- s - scaling layer. Reduces the map size by pooling. Must contain the "scale" field, that is also a vector with 2 integer values. May additionally contain 'stride' field, that determines the step in each dimension. By default is equal to 'scale'.
- f - fully connected layer. Must contain the "length" field that defines the number of its outputs. Must be the last one. For the last layer the length must coincide with the number of classes. May also contain the "dropout" field, that determines the probabilty of dropping the weights on this layer. Thus, it works as DropConnect. Should not be too large, otherwise it drops everything.

All layers except "i" may contain the "function" field, that defines their action. For:
- c and f - it defines the non-linear transformation function. It can be "soft", "sigm" or "relu", for softmax, sigmoid and rectified linear unit respectively. The default value is "relu".
- f - it can also be "SVM", that calculates the SVM error function. For "SVM" the derivatives are not restricted to be in (0, 1), so the learning rate should be about 10 - 100 times smaller than usual.
See [this article](www.cs.toronto.edu/~tang/papers/dlsvm.pdf) for the details. Has been tested only for the final layer.
- s - it defines the pooling procedure, that can be either "mean" or "max". The default value is "mean". 

params - define the learning process. It is a structure with the following fields. If some of them are absent, the value by default is taken.
- seed - any nonnegative integer, that allows to repeat the same random numbers. Default is 0.
- batchsize - defines the size of batches. Default is 50.
- numepochs - the number of repeats the training procedure with different batch splits. Default is 1.
- alpha - defines the learning rate speed. Default is 1. Can aslo be the vector of the length 'numepochs'. Then an individual rate for each epoch will be used.
- momentum - defines the actual direction of weight change according to the formula m * dp + (1-m) * d, where m is momentum, dp is the previous change and d is the current derivative. Default is 0. Can aslo be the vector of the length 'numepochs'. Then an individual momentum for each epoch will be used.
- adjustrate - defines how much we change the learning rate for a particular weight. If the signs of previous and current updates coincide we add it to the learning rate. If not, we divide the learning rate on (1 - adjustrate). Default is 0.
- maxcoef - defines the maximum and minimum learning rates, that are alpha * maxcoef and alpha / maxcoef respectively. Default is 1.
- balance - boolean variable. Balances errors according to the class appearance frequencies. Useful for highly unbalanced datasets. Default is 0.
- shuffle - determines whether the input dataset will be shuffled or not. If it is set to 0, the batches are created in a natural order: first "batchsize" objects become the first batch and so on. Otherwise, it should be 1. Default is 1.
- verbose - determines output info during learning. For 0 there is no output, for 1 it prints only number of epochs, for 2 it prints both numbers of epoch and batch. Default is 2.

weights - the weights vector obtained from genweights or cnntrain, that is used for weights initialization. Can be used for testing or continuing the training procedure. 

funtype - defines the actual function that is used. Can be either "mexfun" or "matlab". "Mexfun" is faster, but in "matlab" it is easier to do some debugging and see the intermediate results.

TECHNICAL DETAILS

- To run the C++ version, you first need to compile it. To do that, you need to run 'compile' script in the main folder. It can be compiled to work with either double and float types. To change it, you need to modify the settings in ftype.h file and recompile the files. 
- You can also specify if you want C++ version to be multi-thread or not by changing the value of macros USE_MULTITHREAD in the same file. By default it is, however if you run several Matlabs in parallel, I would recommend to use the single-thread version.
- For stability purposes all values that are less then a threshold are considered as 0. You can change the threshold in ftype.h and cnnsetup.m, however you need to understand why you do this.

- The code uses c++11 features, so the compiler must understand them. In Windows it was tested with Microsoft SDK 7.1, in Ubuntu with g++ 4.7. Note, that it is possible to use g++ 4.7 only in Matlab R2013b or later. However, it does not understand c++11 by default. To enable c++11 features in Linux, you need to do the following:
1). Copy the settings file <MatlabRoot>/bin/mexopts.sh to ~/.matlab/<MatlabVersion>,
2). Find and change the line "CXXFLAGS='-ansi -D_GNU_SOURCE'" to "CXXFLAGS='-std=c++11 -D_GNU_SOURCE'"
Now you can compile the code.

- Random numbers are used for generating weights, shuffling batches and dropout. Thus, if want to get identical resutls in both Matlab and C++ versions, you need to use the same initial weights, do not use shuffling and set dropout to 0. Note, that these versions produce different random numbers for the same seeds.

SOME COMMENTS 

- The library was developed for Matlab, but probably works in Octave as well. In case the matlab "imdilate" function does not work, you can use the mex-function "maxscale" instead. Just uncomment it in the corresponding block and compile by 'mex maxscale' if necessary.

- In order to achieve compatibility with mex, there are some unnecessary transpose operations in the matlab code. If you do need it, you can remove them.

ACKNOLEDGEMENTS

- The original Matlab code and the "mnist_uint8.mat" workspace was created by [Rasmus Berg Palm](dtu.academia.edu/RasmusBergPalm) and can be found in his [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox). The Matlab version basically remained the same structure as there.

- The C++ version was inspired by [Yichuan Tang](http://www.cs.toronto.edu/~tang) and his [solution](http://code.google.com/p/deep-learning-faces/) for the Kaggle Facial Expression Recognition Challenge. The structure of C++ code was originated from there.
 
