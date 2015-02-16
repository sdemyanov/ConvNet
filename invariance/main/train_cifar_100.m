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
  kDatasetFolder = 'C:/Users/sergeyd/Workspaces/CIFAR-100/data';
  kWorkspaceFolder = 'C:/Users/sergeyd/Workspaces/CIFAR-100';
else
  kDatasetFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/CIFAR-100/data';
  kWorkspaceFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/CIFAR-100';
end;

load(fullfile(kDatasetFolder, 'cifar-100.mat'), 'TrainX', 'TrainY');  
load(fullfile(kDatasetFolder, 'cifar-100.mat'), 'TestX', 'TestY');


kSampleDim = ndims(TrainX);
assert(size(TrainX, kSampleDim) == size(TrainY, 1));
assert(size(TestX, kSampleDim) == size(TestY, 1));
kTrainNum = size(TrainY, 1);
kTestNum = size(TestY, 1);

kXSize = [size(TrainX, 1) size(TrainX, 2)];
kOutputs = size(TrainY, 2);

if (kSampleDim == 3)
  TestX = TestX(:, :, 1:kTestNum);  
  kXSize(3) = 1;
elseif (kSampleDim == 4)
  TestX = TestX(:, :, :, 1:kTestNum);
  kXSize(3) = size(TrainX, 3);
end;
TestY = TestY(1:kTestNum, :);

is_norm = 1;

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
  struct('type', 'c', 'filtersize', [5 5], 'outputmaps', 64, 'padding', [2 2], ...
         'initstd', 0.01, 'biascoef', 2, 'sumwidth', 2)
  struct('type', 's', 'scale', [3 3], 'function', 'max', 'stride', [2 2])
  struct('type', 'c', 'filtersize', [5 5], 'outputmaps', 64, 'padding', [2 2], ...
         'initstd', 0.01, 'biascoef', 2, 'sumwidth', 2, 'unshared', 0)
  struct('type', 'f', 'length', 256, ...
         'initstd', 0.1, 'biascoef', 2)
  struct('type', 'f', 'length', kOutputs, 'function', 'soft', ...
         'initstd', 0.1, 'biascoef', 2)
};

alpha = 0.1;
beta = [0 0 5e-4 5e-5]; 
dropout = [0 0.5 0 0.5];
assert(length(beta) == length(dropout));
factor1 = 0.99;
factor2 = 0.99;

kBetaNum = length(beta);
kIterNum = 10;
kEpochNum = 800;

errors = zeros(kEpochNum, kIterNum, kBetaNum);
losses = zeros(kEpochNum, kIterNum, kBetaNum);
losses2 = zeros(kEpochNum, kIterNum, kBetaNum);
weights = single(genweights(layers, params, funtype));
WeightsIn = zeros(length(weights), kIterNum, kBetaNum);

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

is_load = 1;
if (is_load == 1)
  for bind = 1 : length(beta)
    cursuf = [suffix '-b' num2str(beta(bind))];
    curdrop = dropout(bind);    
    if (curdrop > 0)
      cursuf = [cursuf '-d' num2str(curdrop)];    
    end;
    resultsfile = fullfile(kWorkspaceFolder, ['results' cursuf '.mat']);
    if (exist(resultsfile, 'file'))
      load(resultsfile, 'curerr', 'curloss', 'curloss2', 'curweights');    
      kEpochNumLoad = min(size(curerr, 1), kEpochNum);
      kIterNumLoad = min(size(curerr, 2), kIterNum);      
      errors(1:kEpochNumLoad, 1:kIterNumLoad, bind) = curerr(1:kEpochNumLoad, 1:kIterNumLoad);
      losses(1:kEpochNumLoad, 1:kIterNumLoad, bind) = curloss(1:kEpochNumLoad, 1:kIterNumLoad);
      losses2(1:kEpochNumLoad, 1:kIterNumLoad, bind) = curloss2(1:kEpochNumLoad, 1:kIterNumLoad);
      WeightsIn(:, 1:kIterNumLoad, bind) = curweights(:, 1:kIterNumLoad);    
    end;    
  end;  
  errors(errors == 0.99) = 0;
end;

%alpha = 0.1;

is_valid = 0;

for iter = 1 : kIterNum

rng(iter);  

if (is_valid > 0)  
  train_num = 40000;
  valid_num = 10000;
  [others_ind, train_ind] = crossvalind('LeaveMOut', kTrainNum, train_num);
  train_x = TrainX(:, :, :, train_ind);
  train_y = TrainY(train_ind, :);  
  others_x = TrainX(:, :, :, others_ind);
  others_y = TrainY(others_ind, :);  
  others_num = kTrainNum - train_num;  
  [others_ind, valid_ind] = crossvalind('LeaveMOut', others_num, valid_num);
  test_x = others_x(:, :, :, valid_ind);
  test_y = others_y(valid_ind, :);
else
  train_num = kTrainNum;
  train_x = TrainX;
  train_y = TrainY;
  test_x = TestX;
  test_y = TestY;
end;

if (is_norm == 1) 
  train_mean = mean(train_x, kSampleDim);
  train_x = train_x - repmat(train_mean, [1 1 1 train_num]);
  test_x = test_x - repmat(train_mean, [1 1 1 kTestNum]);  
end;

for bind = 1 : length(beta) 
  
  disp(['Iter: ' num2str(iter)]);
  disp(['Beta: ' num2str(beta(bind))]);
  disp(['Dropout: ' num2str(dropout(bind))]);
  
  curdrop = dropout(bind);
  layers{end-1}.dropout = curdrop;
  
  cursuf = [suffix '-b' num2str(beta(bind))];
  if (curdrop > 0)
    cursuf = [cursuf '-d' num2str(curdrop)];    
  end;

  resultsfile = fullfile(kWorkspaceFolder, ['results' cursuf '.mat']);  

  rng(iter);
  params.seed = iter;  
  first_epoch = find(errors(:, iter, bind) == 0, 1);
  if (first_epoch > 1)
    weights = single(WeightsIn(:, iter, bind));  
  else    
    weights = single(genweights(layers, params, funtype));
  end;
   
  params.alpha = alpha;
  params.beta = beta(bind);  
  for epoch = 1 : first_epoch - 1
    params.alpha = params.alpha * factor1;
    params.beta = params.beta * factor2;  
  end;  
  
  for epoch = first_epoch : kEpochNum
    
    rng(iter * epoch);  
    params.seed = iter * epoch;
    disp(['Epoch: ' num2str(epoch)]);
    
    err = 0.99;
    while (err == 0.99)
      [weights, trainerr] = cnntrain(layers, weights, params, single(train_x), single(train_y), funtype);
      WeightsIn(:, iter, bind) = weights;
      losses(epoch, iter, bind) = trainerr(1);
      %disp([num2str(losses(epoch, iter)) ' loss']);  
      losses2(epoch, iter, bind) = trainerr(2);  
      %disp([num2str(losses2(epoch, iter)) ' loss2']);

      %if (mod(epoch, 5) == 0)
      [err, bad, pred] = cnntest(layers, weights, params, single(test_x), single(test_y), funtype);
      %[err, bad, pred] = multitest(layers, weights, params, single(TestX), single(TestY), funtype);
      %end;
    end;
    errors(epoch, iter, bind) = err;  
    disp([num2str(err * 100) '% error']);
    disp(' ');

    params.alpha = params.alpha * factor1;   
    params.beta = params.beta * factor2;

    if (mod(epoch, 10) == 0)      
      curweights = WeightsIn(:, :, bind);    
      curloss = losses(:, :, bind);
      curloss2 = losses2(:, :, bind);    
      curerr = errors(:, :, bind);
      save(resultsfile, 'curweights', 'curloss', 'curloss2', 'curerr', 'layers', 'params');
    end;
    
    if (epoch > 10 && err > 0.90) 
      % if it diverges, we don't calculate further
      curloss = losses(:, :, bind);
      curloss2 = losses2(:, :, bind);
      curerr = errors(:, :, bind);
      save(resultsfile, 'curweights', 'curloss', 'curloss2', 'curerr', 'layers', 'params');
      break;
    end;   
  end;
  
end;
end;