close all; clear mex;

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
  kDatasetFolder = 'C:/Users/sergeyd/Workspaces/MNIST/data';
  kWorkspaceFolder = 'C:/Users/sergeyd/Workspaces/MNIST';
else
  kDatasetFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/MNIST/data';
  kWorkspaceFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/MNIST';
end;

load(fullfile(kDatasetFolder, 'mnist.mat'), 'TrainX', 'TrainY');  
load(fullfile(kDatasetFolder, 'mnist.mat'), 'TestX', 'TestY');

kSampleDim = ndims(TrainX);
assert(size(TrainX, kSampleDim) == size(TrainY, 1));
assert(size(TestX, kSampleDim) == size(TestY, 1));
kTrainNum = size(TrainY, 1);
kTestNum = size(TestY, 1);

kXSize = [size(TrainX, 1) size(TrainX, 2)];
kOutputs = size(TrainY, 2);

TrainX = TrainX(:, :, 1:kTrainNum);
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
  TrainX = TrainX - repmat(train_mean, [1 1 kTrainNum]);
  TestX = TestX - repmat(train_mean, [1 1 kTestNum]);  
  
end;

params.epochs = 1;
%params.epochstest = 1;
params.momentum = 0;  
params.lossfun = 'logreg';
params.shuffle = 1;
params.verbose = 0;
params.seed = 0;

layers = {
  struct('type', 'i', 'mapsize', kXSize(1:2), 'outputmaps', kXSize(3))
  struct('type', 'j', 'mapsize', [28 28], 'shift', [2 2], ...
         'scale', [1.4 1.4], 'angle', 0.1, 'defval', 0)
  struct('type', 'c', 'filtersize', [4 4], 'outputmaps', 32, 'padding', [0 0])
  struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2])
  struct('type', 'c', 'filtersize', [5 5], 'outputmaps', 64, 'padding', [2 2])
  struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2])
  struct('type', 'f', 'length', 256)
  struct('type', 'f', 'length', kOutputs, 'function', 'soft')
};

alpha = 0.1;
beta = [0 2e-3 0 2e-4];
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

stdev = 0 : 0.005 : 0.1;
kStNum = length(stdev);
kIterNum = 10;

errors = zeros(kStNum, kIterNum, kBetaNum);    
clear legstr;

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
      test_y = TestY;  
      [err, bad, pred] = cnntest(layers, single(weights), params, single(test_x), single(test_y), funtype);
      errors(stind, iter, bind) = err;
    end;
    
  end;  
end;
%%
errfile = fullfile(kWorkspaceFolder, 'noise.mat');
save(errfile, 'errors', 'stdev');



  

