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

kTestBatchNum = 3;
cursize = 0;
TestX = single([]);
TestY = single([]);
for bind = 1 : kTestBatchNum
  batch_name = ['test-batch-' num2str(bind) '.mat'];
  load(fullfile(kDatasetFolder, batch_name), 'batch_x', 'batch_y');    
  curind = cursize + 1 : cursize + size(batch_y, 1);
  TestX(:, :, :, curind) = batch_x;
  TestY(curind, :) = batch_y;
  cursize = cursize + size(batch_y, 1);
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
beta = [0 0 0.01];
dropout = [0 0.5 0.5];
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

stdev = 0 : 0.02 : 0.4;
kStNum = length(stdev);
kIterNum = 4;

%errors = zeros(kStNum, kIterNum, kBetaNum);    
clear legstr;

for iter = 2 : kIterNum
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
  
  errfile = fullfile(kWorkspaceFolder, 'noise.mat');
  save(errfile, 'errors', 'stdev');
  
end;



  

