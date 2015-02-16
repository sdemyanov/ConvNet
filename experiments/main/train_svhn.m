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
  kDatasetFolder = 'C:/Users/sergeyd/Workspaces/SVHN/data';
  kWorkspaceFolder = 'C:/Users/sergeyd/Workspaces/SVHN';
else
  kDatasetFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/SVHN/data';
  kWorkspaceFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/SVHN';
end;

kXSize = [32 32 3];
kOutputs = 10;

is_norm = 0;

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
         'scale', [1.4 1.4], 'angle', 0.0, 'defval', 0)
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
  struct('type', 'f', 'length', 256, ...
         'initstd', 0.1, 'biascoef', 2)    
  struct('type', 'f', 'length', kOutputs, 'function', 'soft', ...
         'initstd', 0.1, 'biascoef', 2)
};

alpha = 0.1;
beta = [0 0 0.01 0.001];
dropout = [0 0.5 0 0.5];
assert(length(beta) == length(dropout));
factor1 = 0.90;
factor2 = 0.90;

kBetaNum = length(beta);
kIterNum = 10;
kEpochNum = 80;

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
      kIterNumLoad = min(size(curerr, 2), kIterNum);
      kEpochNumLoad = min(size(curerr, 1), kEpochNum);
      errors(1:kEpochNumLoad, 1:kIterNumLoad, bind) = curerr(1:kEpochNumLoad, 1:kIterNumLoad);
      losses(1:kEpochNumLoad, 1:kIterNumLoad, bind) = curloss(1:kEpochNumLoad, 1:kIterNumLoad);
      losses(1:kEpochNumLoad, 1:kIterNumLoad, bind) = curloss2(1:kEpochNumLoad, 1:kIterNumLoad);
      WeightsIn(:, 1:kIterNumLoad, bind) = curweights(:, 1:kIterNumLoad);
    end;
  end;  
end;

is_valid = 0;

for iter = 10 : kIterNum

rng(iter);
  
batchsize = 10000;
kTotalBatchNum = 60;

if (is_valid > 0)  
  kTrainBatchNum = 6;
  batch_name = 'valid-batch-1.mat';
  load(fullfile(kDatasetFolder, batch_name), 'batch_x', 'batch_y');
  test_x = batch_x;
  test_y = batch_y;
else
  kTrainBatchNum = 60;
  kTestBatchNum = 3;
  cursize = 0;
  test_x = single([]);
  test_y = single([]);
  for bind = 1 : kTestBatchNum
    batch_name = ['test-batch-' num2str(bind) '.mat'];
    load(fullfile(kDatasetFolder, batch_name), 'batch_x', 'batch_y');    
    curind = cursize + 1 : cursize + size(batch_y, 1);
    test_x(:, :, :, curind) = batch_x;
    test_y(curind, :) = batch_y;
    cursize = cursize + size(batch_y, 1);
  end;  
end;

[others_ind, train_batches] = crossvalind('LeaveMOut', kTotalBatchNum, kTrainBatchNum);
train_ind = find(train_batches);
cursize = 0;
train_x = single([]);
train_y = single([]);
for bind = 1 : kTrainBatchNum
  batch_name = ['train-batch-' num2str(train_ind(bind)) '.mat'];
  load(fullfile(kDatasetFolder, batch_name), 'batch_x', 'batch_y');
  curind = cursize + 1 : cursize + size(batch_y, 1);
  train_x(:, :, :, curind) = batch_x;
  train_y(curind, :) = batch_y;
  cursize = cursize + size(batch_y, 1);
end;

for bind = 1 : length(beta) 
  
  disp(['Iter: ' num2str(iter)]);
  disp(['Beta: ' num2str(beta(bind))]);
  disp(['Dropout: ' num2str(dropout(bind))]);
  
  curdrop = dropout(bind);
  layers{end - 1}.dropout = curdrop;
  
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
    
    err = 0.90;
    while (err == 0.90)
      [weights, trainerr] = cnntrain(layers, weights, params, single(train_x), single(train_y), funtype);
      WeightsIn(:, iter, bind) = weights;
      losses(epoch, iter, bind) = trainerr(1);
      %disp([num2str(losses(epoch, iter)) ' loss']);  
      losses2(epoch, iter, bind) = trainerr(2);  
      %disp([num2str(losses2(epoch, iter)) ' loss2']);
      
      [err, bad, pred] = cnntest(layers, weights, params, single(test_x), single(test_y), funtype);      
    end;
    
    %end;
    errors(epoch, iter, bind) = err;  
    disp([num2str(err * 100) '% error']);
    disp(' ');

    params.alpha = params.alpha * factor1;   
    params.beta = params.beta * factor2;

    if (mod(epoch, 5) == 0)
      curweights = WeightsIn(:, :, bind);
      curloss = losses(:, :, bind);
      curloss2 = losses2(:, :, bind);
      curerr = errors(:, :, bind);
      save(resultsfile, 'curweights', 'curloss', 'curloss2', 'curerr', 'layers', 'params');
    end;
    
    if (epoch > 10 && err > 0.85) 
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
