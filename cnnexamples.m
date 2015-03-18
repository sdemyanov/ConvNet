
close all; clear mex;

cd(fileparts(mfilename('fullpath')));

funtype = 'gpu';
%funtype = 'cpu';
%funtype = 'matlab';

disp(funtype);

if (strcmp(funtype, 'gpu') || strcmp(funtype, 'cpu'))  
  kMexFolder = './c++/build';
  kBuildFolder = fullfile(kMexFolder, funtype);
  if (ispc)
    mexfiles = fullfile(kBuildFolder, '*.mexw64');
  elseif (isunix)  
    mexfiles = fullfile(kBuildFolder, '*.mexa64');
  end;
  copyfile(mexfiles, kMexFolder, 'f');
  addpath(kMexFolder);
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
%kTrainNum = 2000;
kOutputs = size(TrainY, 2);
train_x = single(TrainX(:, :, 1:kTrainNum));
train_y = single(TrainY(1:kTrainNum, :));

kTestNum = 10000;
test_x = single(TestX(:, :, 1:kTestNum));
test_y = single(TestY(1:kTestNum, :));

clear params;
params.epochs = 1;
params.alpha = 0.1;
% this is the parameter for invariant backpropagation
% keep it 0 for standard backpropagation
params.beta = 0; 
params.momentum = 0.9;
params.lossfun = 'logreg';
params.shuffle = 1;
params.seed = 0;
dropout = 0;

% norm_x = squeeze(mean(sqrt(sum(sum(train_x.^2))), kSampleDim));

% !!! IMPORTANT NOTICES FOR GPU VERSION !!!
% Outputmaps number should be divisible on 16
% For speed use only the default value of batchsize = 128

% This structure gives pretty good results on MNIST after just several epochs

layers = {
  struct('type', 'i', 'mapsize', kXSize(1:2), 'outputmaps', kXSize(3))
  % remove the following layer in the Matlab version - it is not implemented there
  struct('type', 'j', 'mapsize', [28 28], 'shift', [1 1], ...
         'scale', [1.40 1.40], 'angle', 0.10, 'defval', 0)  
  struct('type', 'c', 'filtersize', [4 4], 'outputmaps', 32)
  struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2])
  struct('type', 'c', 'filtersize', [5 5], 'outputmaps', 64, 'padding', [2 2])
  struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2])
  struct('type', 'f', 'length', 256, 'dropout', dropout)
  struct('type', 'f', 'length', kOutputs, 'function', 'soft')
};

weights = single(genweights(layers, params, funtype));
EpochNum = 10;
errors = zeros(EpochNum, 1);
for i = 1 : EpochNum
  disp(['Epoch: ' num2str((i-1) * params.epochs + 1)])
  [weights, trainerr] = cnntrain(layers, weights, params, train_x, train_y, funtype);  
  disp([num2str(mean(trainerr(:, 1))) ' loss']);  
  [err, bad, pred] = cnntest(layers, weights, params, test_x, test_y, funtype);  
  disp([num2str(err*100) '% error']);  
  errors(i) = err;
  params.alpha = params.alpha * 0.95;
  params.beta = params.beta * 0.95;
end;
plot(errors);
disp('Done!');

