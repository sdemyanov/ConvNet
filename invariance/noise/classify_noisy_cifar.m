close all; clear mex;

%datasetName = 'CIFAR';
datasetName = 'CIFAR-100';

funtype = 'gpu';
%funtype = 'cpu';
%funtype = 'matlab';

kCNNFolder = '..';
addpath(kCNNFolder);
addpath(fullfile(kCNNFolder, 'matlab'));

if (strcmp(funtype, 'gpu') || strcmp(funtype, 'cpu'))  
  kMexFolder = fullfile(kCNNFolder, 'c++', 'build');  
  
  kBuildFolder = fullfile(kMexFolder, funtype);
  if (strcmp(funtype, 'gpu'))
    if (ispc)
      kGPUFolder = 'C:\Users\sergeyd\Documents\Visual Studio 2012\Projects\mexfiles\x64\Release';    
      copyfile(fullfile(kGPUFolder, '*.mexw64'), kBuildFolder, 'f');    
    end;
  end;  
  copyfile(fullfile(kBuildFolder, '*'), kMexFolder, 'f');  
  addpath(kMexFolder);
end;

if (ispc)
  kDatasetFolder = fullfile('C:/Users/sergeyd/Workspaces', datasetName, 'data');
  kWorkspaceFolder = fullfile('C:/Users/sergeyd/Workspaces', datasetName);
else
  kDatasetFolder = fullfile('/media/sergeyd/OS/Users/sergeyd/Workspaces', datasetName, 'data');
  kWorkspaceFolder = fullfile('/media/sergeyd/OS/Users/sergeyd/Workspaces', datasetName);
end;

if (strcmp(datasetName, 'CIFAR'))
  dsfile = 'cifar.mat';
elseif (strcmp(datasetName, 'CIFAR-100'))
  dsfile = 'cifar-100.mat';
end;  

load(fullfile(kDatasetFolder, dsfile), 'TrainX', 'TrainY');  
load(fullfile(kDatasetFolder, dsfile), 'TestX', 'TestY');

kWorkspaceFolder = fullfile(kWorkspaceFolder, 'final-last');

kSampleDim = ndims(TrainX);
assert(size(TrainX, kSampleDim) == size(TrainY, 1));
assert(size(TestX, kSampleDim) == size(TestY, 1));
kTrainNum = size(TrainY, 1);
kTestNum = size(TestY, 1);

kXSize = [size(TrainX, 1) size(TrainX, 2)];
kOutputs = size(TrainY, 2);

TrainX = TrainX(:, :, :, 1:kTrainNum);
TrainY = TrainY(1:kTrainNum, :);
if (kSampleDim == 3)
  TestX = TestX(:, :, 1:kTestNum);  
  kXSize(3) = 1;
elseif (kSampleDim == 4)
  TestX = TestX(:, :, :, 1:kTestNum);
  kXSize(3) = size(TrainX, 3);
end;
TestY = TestY(1:kTestNum, :);

is_norm = 1;
if (is_norm == 1)  
  train_mean = mean(TrainX, kSampleDim);
  TrainX = TrainX - repmat(train_mean, [1 1 1 kTrainNum]);
  TestX = TestX - repmat(train_mean, [1 1 1 kTestNum]);  
end;

params.epochs = 1;
%params.epochstest = 1;
params.momentum = 0.9;  
params.lossfun = 'logreg';
params.shuffle = 1;
params.verbose = 0;
params.seed = 0;

layers = {
  struct('type', 'i', 'mapsize', kXSize(1:2), 'outputmaps', kXSize(3))
  %struct('type', 'j', 'mapsize', [28 28], 'shift', [4 4], 'defval', 0)
  struct('type', 'j', 'mapsize', [28 28], 'shift', [2 2], ...
         'scale', [1.4 1.4], 'mirror', [0 1], 'angle', 0.0, 'defval', 0)
  struct('type', 'c', 'filtersize', [5 5], 'outputmaps', 64, 'padding', [0 0], ...
         'initstd', 0.01, 'biascoef', 2, 'sumwidth', 4)
  struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2])
  %struct('type', 'n', 'normsize', 9, 'scale', 0.001, 'pow', 0.75)
  struct('type', 'c', 'filtersize', [5 5], 'outputmaps', 64, 'padding', [2 2], ...
         'initstd', 0.01, 'biascoef', 2, 'sumwidth', 2)
  %struct('type', 'n', 'normsize', 9, 'scale', 0.001, 'pow', 0.75)
  struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2])
  struct('type', 'c', 'filtersize', [5 5], 'outputmaps', 64, 'padding', [2 2], ...
         'initstd', 0.01, 'biascoef', 2, 'sumwidth', 2, 'unshared', 0)
  struct('type', 'f', 'length', 256, ...
         'initstd', 0.1, 'biascoef', 2)
  struct('type', 'f', 'length', kOutputs, 'function', 'soft', ...
         'initstd', 0.1, 'biascoef', 2)  
};

alpha = 0.1;
beta = [0 5e-4 0 5e-5];
dropout = [0 0 0.5 0.5];

assert(length(beta) == length(dropout));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kBetaNum = length(beta);

suffix = '';
if (is_norm == 1)
  suffix = [suffix '-n'];
  disp('Normalized input');  
end;
if (params.momentum > 0)
  suffix = [suffix '-m' num2str(params.momentum)];    
  disp(['Momentum: ' num2str(params.momentum)]);
end;
if (alpha ~= 1)
  suffix = [suffix '-a' num2str(alpha)];    
  disp(['Alpha: ' num2str(alpha)]);
end;

stdev = 0 : 0.005 : 0.2;
kStNum = length(stdev);
kIterNum = 2;

errors = zeros(kStNum, kIterNum, kBetaNum);    
clear legstr;

test_y = single(TestY);

for iter = 1 : kIterNum
  for bind = 1 : kBetaNum
    
    disp(['bind = ' num2str(bind) ', iter = ' num2str(iter)]);    
    
    cursuf = [suffix '-b' num2str(beta(bind))];
    legstr{bind} = cursuf;
    curdrop = dropout(bind);    
    if (curdrop > 0)
      cursuf = [cursuf '-d' num2str(curdrop)];    
    end;
    resultsfile = fullfile(kWorkspaceFolder, ['results' cursuf '.mat']);
    if (exist(resultsfile, 'file'))
      load(resultsfile, 'curweights');    
      WeightsIn = curweights;    
    end;
  
    weights = WeightsIn(:, iter);    
    for stind = 1 : kStNum
      rng(iter);      
      test_x = TestX + stdev(stind) * single(randn(size(TestX)));
      [err, bad, pred] = cnntest(layers, single(weights), params, single(test_x), single(test_y), funtype);
      errors(stind, iter, bind) = err;
    end;
    
  end;  
end;
%%
errfile = fullfile(kWorkspaceFolder, ['noise' num2str(max(stdev)) '.mat']);
save(errfile, 'errors', 'stdev');
  

