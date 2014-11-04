
close all; clear mex;

cd(fileparts(mfilename('fullpath')));

funtype = 'gpu';
%funtype = 'cpu';
%funtype = 'matlab';

disp(funtype);

if (strcmp(funtype, 'gpu') || strcmp(funtype, 'cpu'))  
  kBuildFolder = './c++/build';
  copyfile(fullfile(kBuildFolder, funtype, '*.mexw64'), kBuildFolder, 'f');
  addpath(kBuildFolder);
end;
addpath('./matlab');
addpath('./data');
load mnist;

kSampleDim = ndims(TrainX);
kXSize = size(TrainX);
kXSize(kSampleDim) = [];
if (kSampleDim == 3)  
  kXSize(3) = 1;
end;
kWorkspaceFolder = './workspace';
if (~exist(kWorkspaceFolder, 'dir'))
  mkdir(kWorkspaceFolder);
end;

kTrainNum = 60000;
%kTrainNum = 12800;
%kTrainNum = 200;
kOutputs = size(TrainY, 2);
train_x = single(TrainX(:, :, 1:kTrainNum));
train_y = single(TrainY(1:kTrainNum, :));

kTestNum = 10000;
test_x = single(TestX(:, :, 1:kTestNum));
test_y = single(TestY(1:kTestNum, :));

clear params;
params.seed = 0;
params.numepochs = 1;
params.alpha = 1;
params.momentum = 0.9;
params.shuffle = 0;
dropout = 0;

norm_x = squeeze(mean(sqrt(sum(sum(train_x.^2))), kSampleDim));

% !!! IMPORTANT NOTICES FOR GPU VERSION !!!
% Outputmaps number should be divisible on 16
% Use only the default value of batchsize = 128

% This structure is just for demonstration purposes

layers = {
    struct('type', 'i', 'mapsize', kXSize(1:2), 'outputmaps', kXSize(3), ...
           'norm', norm_x, 'mean', 0', 'maxdev', 1)
    struct('type', 'c', 'filtersize', [4 4], 'outputmaps', 16, 'padding', [1 1]) %convolution layer
    struct('type', 's', 'scale', [3 3], 'function', 'mean', 'stride', [2 2]) % subsampling layer
    struct('type', 'c', 'filtersize', [4 4], 'outputmaps', 32, 'padding', [1 1]) %convolution layer
    struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2]) % subsampling layer   
    struct('type', 'f', 'length', 256, 'dropout', dropout) % fully connected layer
    struct('type', 'f', 'length', kOutputs, 'function', 'soft') % fully connected layer
};

rng(params.seed);
weights = single(genweights(layers, params.seed, 'matlab'));
EpochNum = 1;
errors = zeros(EpochNum, 1);
for i = 1 : EpochNum
  disp(['Epoch: ' num2str((i-1) * params.numepochs) + 1])
  [weights, trainerr] = cnntrain(layers, weights, params, train_x, train_y, funtype);  
  disp([num2str(mean(trainerr(:, 1))) ' loss']);  
  [err, bad, pred] = cnntest(layers, weights, params, test_x, test_y, funtype);  
  disp([num2str(err*100) '% error']);  
  errors(i) = err;
end;
%plot(errors);
disp('Done!');

