
close all; clear mex;

addpath('./c++/build');
addpath('./matlab');
addpath('./data');
load mnist;

kSampleDim = ndims(TrainX);
kXSize = size(TrainX);
kXSize(kSampleDim) = [];
kWorkspaceFolder = './workspace';
if (~exist(kWorkspaceFolder, 'dir'))
  mkdir(kWorkspaceFolder);
end;

kTrainNum = 10000;
kOutputs = size(TrainY, 2);
train_x = TrainX(:, :, 1:kTrainNum);
train_y = TrainY(1:kTrainNum, :);

kTestNum = 10000;
test_x = TestX(:, :, 1:kTestNum);
test_y = TestY(1:kTestNum, :);

params.seed = 1;
params.batchsize = 50;
params.numepochs = 1;
params.alpha = 1;
params.momentum = 0.9;
params.shuffle = 0;
params.verbose = 0;
dropout = 0;

norm_x = squeeze(mean(sqrt(sum(sum(train_x.^2))), kSampleDim));

% This structure is just supposed to demonstrate the implemented options
layers = {
    struct('type', 'i', 'mapsize', kXSize, 'outputmaps', 1, ...
           'norm', norm_x, 'mean', 0', 'maxdev', 1);
    struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 6) %convolution layer
    struct('type', 's', 'scale', [3 3], 'function', 'mean', 'stride', [2 2]) % subsampling layer
    struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 12, 'padding', [1 1]) %convolution layer
    struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2]) % subsampling layer        
    struct('type', 'f', 'length', 64) % fully connected layer
    struct('type', 'f', 'length', kOutputs, 'function', 'soft', ...
           'dropout', dropout) % fully connected layer
};

funtype = 'mexfun';
%funtype = 'matlab';

rng(params.seed);
weights_in = genweights(layers, params.seed, 'matlab');
EpochNum = 1;
errors = zeros(EpochNum, 1);
weights = weights_in;
for i = 1 : EpochNum
  disp(['Epoch: ' num2str(i)])
  [weights, trainerr] = cnntrain(layers, weights, train_x, train_y, params, funtype);  
  plot(trainerr);
  disp([num2str(mean(trainerr)) ' loss']);
  [err, bad, pred]  = cnntest(layers, weights, test_x, test_y, funtype);  
  disp([num2str(err*100) '% error']);
  errors(i) = err;
end;

%save(fullfile(kWorkspaceFolder, 'weights.mat'), 'weights');  
