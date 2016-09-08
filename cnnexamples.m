
close all; clear mex;

restoredefaultpath;
kCNNFolder = '.';  
kMexFolder = fullfile(kCNNFolder, 'c++', 'build');
addpath(kMexFolder);

addpath('./matlab');
addpath('./data');
load mnist_uint8;
TrainX = single(permute(reshape(train_x', [28 28 1 60000]), [2 1 3 4])) / 255;
TestX = single(permute(reshape(test_x', [28 28 1 10000]), [2 1 3 4])) / 255;
TrainY = single(permute(train_y, [3 4 2 1]));
TestY = single(permute(test_y, [3 4 2 1]));

kXSize = size(TrainX);
kXSize(end+1:4) = 1;
kOutputs = size(TrainY, 3);
train_mean = mean(mean(mean(TrainX, 1), 2), 4);
TrainX = TrainX - repmat(train_mean, [kXSize(1) kXSize(2) 1 size(TrainX, 4)]);
TestX = TestX - repmat(train_mean, [kXSize(1) kXSize(2) 1 size(TestX, 4)]);

kWorkspaceFolder = './workspace';
if (~exist(kWorkspaceFolder, 'dir'))
  mkdir(kWorkspaceFolder);
end;

kTrainNum = 60000;
train_x = single(TrainX(:, :, :, 1:kTrainNum));
train_y = single(TrainY(:, :, :, 1:kTrainNum));

kTestNum = 10000;
test_x = single(TestX(:, :, :, 1:kTestNum));
test_y = single(TestY(:, :, :, 1:kTestNum));

clear params;
params.epochs = 1;
params.alpha = 0.1;
params.beta = 0;
params.momentum = 0;
params.lossfun = 'logreg';
params.shuffle = 1;
params.seed = 1;
dropout = 0;
params.balance = 0;
params.verbose = 0;
params.gpu = 1;
gamma = 0.95;

layers = {
  struct('type', 'input', 'mapsize', kXSize(1:2), 'channels', kXSize(3))
  struct('type', 'jitt', 'mapsize', kXSize(1:2), 'shift', [3 3], ...
         'scale', [1.2 1.2], 'angle', 0.1, 'defval', -train_mean) 
  struct('type', 'conv', 'filtersize', [4 4], 'channels', 32)
  struct('type', 'pool', 'scale', [3 3], 'stride', [2 2])
  struct('type', 'conv', 'filtersize', [5 5], 'channels', 64)
  struct('type', 'pool', 'scale', [3 3], 'stride', [2 2])
  struct('type', 'full', 'channels', kOutputs, 'function', 'soft')
};

layers = setup(layers);
layers = genweights(layers, params, 'matlab');

EpochNum = 3;
errors = zeros(EpochNum, 1);

for i = 1 : EpochNum
  disp(['Epoch: ' num2str((i-1) * params.epochs + 1)]);
  
  [layers, trainerr] = train(layers, params, train_x, train_y);  
  disp([num2str(mean(trainerr(:, 1))) ' loss']);  
  [err, bad, pred_y] = test(layers, params, test_x, test_y);  
  disp([num2str(err*100) '% error']);
  errors(i) = err;
  params.alpha = params.alpha * gamma;
  params.beta = params.beta * gamma;
  disp('');
  
end;
disp('Done!');
