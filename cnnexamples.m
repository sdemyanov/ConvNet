
close all; clear mex;
clear;

addpath('./c++/build');
addpath('./matlab');
addpath('./data');
load mnist_uint8;

kXSize = [28 28];
train_x = double(permute(reshape(train_x', [kXSize size(train_x, 1)]), [2 1 3]))/255;
test_x = double(permute(reshape(test_x', [kXSize, size(test_x, 1)]), [2 1 3]))/255;
train_y = double(train_y);
test_y = double(test_y);

kWorkspaceFolder = './workspace';
if (~exist(kWorkspaceFolder, 'dir'))
  mkdir(kWorkspaceFolder);
end;

kTrainNum = 60000;
kOutputs = size(train_y, 2);
train_x = train_x(:, :, 1:kTrainNum);
train_y = double(train_y(1:kTrainNum, :));

kTestNum = 10000;
test_x = test_x(:, :, 1:kTestNum);
test_y = double(test_y(1:kTestNum, :));

mean_s = mean(mean(train_x, 1), 2);
train_x_unbiased = train_x - repmat(mean_s, [kXSize 1]);
norm_x = 2*mean(squeeze(sqrt(sum(sum(train_x_unbiased.^2)))));
kMinVar = 1;
mean_x = mean(train_x, 3);
std_x = sqrt(var(train_x, 0, 3) + kMinVar);

params.batchsize = 50;
params.numepochs = 2;
params.alpha = [1 0.9];
params.momentum = [0.5 0.9];  
params.shuffle = 0;
params.verbose = 2;

layers = {
    %struct('type', 'i', 'mapsize', kXSize, 'outputmaps', 1)
    struct('type', 'i', 'mapsize', kXSize, 'outputmaps', 1, ...
           'norm', norm_x, 'mean', mean_x, 'stdev', std_x)    
    struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 6, 'function', 'relu') %convolution layer
    struct('type', 's', 'scale', [3 3], 'function', 'mean', 'stride', [2 2]) % subsampling layer
    struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 12, 'padding', [1 1]) %convolution layer
    struct('type', 's', 'scale', [2 2], 'function', 'max') % subsampling layer        
    struct('type', 'f', 'length', kOutputs, 'function', 'soft', 'dropout', 0) % fully connected layer
};

funtype = 'mexfun';
%funtype = 'matlab';
weights_in = genweights(layers, funtype);
save(fullfile(kWorkspaceFolder, 'weights_in.mat'), 'weights_in');
%load(fullfile(kWorkspaceFolder, 'weights_in.mat'), 'weights_in');
EpochNum = 1;
errors = zeros(EpochNum, 1);
weights = weights_in;
for i = 1 : EpochNum
  disp(['Number: ' num2str(i)])
  [weights, trainerr] = cnntrain(layers, weights, train_x, train_y, params, funtype);  
  plot(trainerr);  
  %save(fullfile(kWorkspaceFolder, 'weights.mat'), 'weights');  
  [err, bad, pred]  = cnntest(layers, weights, test_x, test_y, funtype);  
  disp([num2str(err*100) '% error']);
  errors(i) = err;
end;
