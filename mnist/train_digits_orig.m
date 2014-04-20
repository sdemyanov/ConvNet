
close all; clear mex;

kWorkspaceFolder = '..';

kCNNFolder = '..';
addpath(kCNNFolder);
addpath(fullfile(kCNNFolder, 'c++/build/'));
addpath(fullfile(kCNNFolder, 'matlab'));

is_ext = 0;
is_rest = 0;
if (is_ext == 0)
  if (is_rest == 0)
    load(fullfile(kWorkspaceFolder, 'data', 'mnist.mat'), 'TrainX', 'TrainY');  
  elseif (is_rest == 1)
    load(fullfile(kWorkspaceFolder, 'data', 'mnist_res.mat'), 'TrainX', 'TrainY');
  end;
  load(fullfile(kWorkspaceFolder, 'data', 'mnist.mat'), 'TestX', 'TestY');
elseif (is_ext == 1)
  if (is_rest == 0)
    load(fullfile(kWorkspaceFolder, 'data', 'mnist_ext.mat'), 'TrainX', 'TrainY');  
  elseif (is_rest == 1)
    load(fullfile(kWorkspaceFolder, 'data', 'mnist_ext_res.mat'), 'TrainX', 'TrainY');
  end;
  load(fullfile(kWorkspaceFolder, 'data', 'mnist_ext.mat'), 'TestX', 'TestY');
end;
kOutputs = size(TrainY, 2);

kMaxTrainNum = size(TrainY, 1);
kTrainNum = 200;
%TrainX = TrainX(:, :, 1:kTrainNum);
%TrainY = double(TrainY(1:kTrainNum, :));

kTestNum = 3000;
TestX = TestX(:, :, 1:kTestNum);
TestY = double(TestY(1:kTestNum, :));

kXSize = [size(TrainX, 1) size(TrainX, 2)];

params.batchsize = 50;
params.numepochs = 1;
params.momentum = 0.9;  
params.shuffle = 0;
params.verbose = 0;

layers = {
  %struct('type', 'i', 'mapsize', kXSize, 'outputmaps', 1, ...
  %       'norm', norm_x, 'mean', mean_x, 'stdev', std_x) 
  struct('type', 'i', 'mapsize', kXSize, 'outputmaps', 1)
  %struct('type', 'f', 'length', 256)
  %struct('type', 'f', 'length', 256)
  struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 6)
  struct('type', 's', 'scale', [2 2], 'function', 'mean')
  struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 12)
  struct('type', 's', 'scale', [2 2], 'function', 'mean')
  struct('type', 'f', 'length', kOutputs, 'function', 'soft')
};

funtype = 'mexfun';
%funtype = 'matlab';

alpha = 0.5;
factor = 0.98;

is_load = 0;
if (is_load == 0)
  kIterNum = 20;
  kEpochNum = 200;
  errors = zeros(kEpochNum, kIterNum);
  losses = zeros(kEpochNum, kIterNum);
  weights_in = genweights(layers, 'matlab');
  WeightsIn = zeros(length(weights_in), kIterNum);
else
  load(fullfile('./results', resultsfile), 'errors', 'losses');
  kIterNum = size(errors, 2);
  kEpochNum = size(errors, 1);
  load(fullfile('./results', weightsfile), 'WeightsIn');
end;  
first_zero = find(errors(:) == 0, 1);
first_iter = floor((first_zero-1) / kEpochNum) + 1;
first_epoch = mod((first_zero-1), kEpochNum) + 1;

for epoch = 1 : first_epoch - 1
  perm_ind = randperm(kTrainNum);    
  params.alpha = params.alpha * factor;
  params.beta = params.beta * factor;  
end;

for iter = first_iter : kIterNum
  
rng(iter);
%perm_ind = randperm(kMaxTrainNum);
%TrainX = TrainX(:, :, perm_ind);
%TrainY = TrainY(perm_ind, :);
%[~, train_ind] = crossvalind('LeaveMOut', kMaxTrainNum, kTrainNum);
%train_x = TrainX(:, :, train_ind);
%train_y = double(TrainY(train_ind, :));
train_x = TrainX(:, :, (iter-1)*kTrainNum+1 : iter*kTrainNum);
train_y = double(TrainY((iter-1)*kTrainNum+1 : iter*kTrainNum, :));

disp(['Iter: ' num2str(iter)])
  
params.alpha = alpha;

if (is_rest == 0)
  currentfile = 'weights-b-cur.mat';
  weightsfile = 'weights-b.mat';
  resultsfile = 'results-b.mat';  
  disp('Beta: baseline');  
else
  currentfile = 'weights-0-cur.mat';
  weightsfile = 'weights-0.mat';
  resultsfile = 'results-0.mat';  
  disp('Beta: 0');
end;
if (is_ext == 1)
  disp('Extended database');  
end;

weights_in = genweights(layers, 'matlab');
%save(fullfile(kWorkspaceFolder, 'weights_in.mat'), 'weights_in');
if (is_load == 1)
  load(fullfile('./results', currentfile), 'weights_in');
end;
weights = weights_in;

for epoch = first_epoch : kEpochNum
  
  perm_ind = randperm(kTrainNum);  
  train_x = train_x(:, :, perm_ind);
  train_y = train_y(perm_ind, :);
  disp(['Epoch: ' num2str(epoch)])
  [weights, trainerr] = cnntrain(layers, weights, train_x, train_y, params, funtype);  
  losses(epoch, iter) = mean(trainerr(:, 1));
  disp([num2str(losses(epoch, iter)) ' loss']);
  %plot(trainerr(:,1));
  
  [err, bad, pred] = cnntest(layers, weights, TestX, TestY, funtype);
  errors(epoch, iter) = err;  
  disp([num2str(errors(epoch, iter)*100) '% error']);
  
  weights_in = weights;  
  save(fullfile('./results', currentfile), 'weights_in', 'trainerr');
  save(fullfile('./results', resultsfile), 'errors', 'losses');    
  params.alpha = params.alpha * factor;   
  disp(' ');
end;

WeightsIn(:, iter) = weights;
save(fullfile('./results', weightsfile), 'WeightsIn', 'trainerr');

end;

%worig = weights;
%disp(sum(abs(worig - wmod)));
