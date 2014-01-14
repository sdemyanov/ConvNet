
close all; clear mex;

addpath('./c++/build');
addpath('./matlab');
addpath('./data');
load mnist_uint8;

kWorkspaceFolder = './workspace';
if (~exist(kWorkspaceFolder, 'dir'))
  mkdir(kWorkspaceFolder);
end;

%load(fullfile(kWorkspaceFolder, 's.mat'), 's');
%rng(s);

kXSize = [28 28];
kTrainNum = 60000;
kTestNum = 10000;
train_x = double(permute(reshape(train_x', [kXSize kTrainNum]), [2 1 3])/255);
test_x = double(permute(reshape(test_x', [kXSize, kTestNum]), [2 1 3])/255);
train_y = double(train_y);
test_y = double(test_y);
kOutputs = size(train_y, 2);

kTrainNum = 5000;
train_x = train_x(:, :, 1:kTrainNum);
train_y = train_y(1:kTrainNum, :);

mean_s = mean(mean(train_x, 1), 2);
train_x_unbiased = train_x - repmat(mean_s, [kXSize 1]);
norm_x = 2*mean(squeeze(sqrt(sum(sum(train_x_unbiased.^2)))));
kMinVar = 1;
mean_x = mean(train_x, 3);
std_x = sqrt(var(train_x, 0, 3) + kMinVar);
%std_x = ones(kXSize);

params.alpha = 1;
params.batchsize = 50;
params.numepochs = 1;
params.momentum = 0;  
params.adjustrate = 0;
params.maxcoef = 10;
params.balance = 0;
params.shuffle = 0;
params.verbose = 2;

layers = {
    struct('type', 'i', 'mapsize', [28 28], 'outputmaps', 1, ...
           'norm', norm_x, 'mean', mean_x, 'stdev', std_x) % input layer    
    struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 6, 'function', 'relu') %convolution layer
    struct('type', 's', 'scale', [3 3], 'function', 'mean', 'stride', [2 2]) % subsampling layer
    struct('type', 'c', 'kernelsize', [5 5], 'outputmaps', 12, 'function', 'relu', 'padding', [1 1]) %convolution layer
    struct('type', 's', 'scale', [2 2], 'function', 'max') % subsampling layer        
    struct('type', 'f', 'length', kOutputs) % fully connected layer
};

funtype = 'mexfun';
%funtype = 'matlab';
weights_in = genweights(layers, funtype);
save(fullfile(kWorkspaceFolder, 'weights_in.mat'), 'weights_in');
%load(fullfile(kWorkspaceFolder, 'weights_in.mat'), 'weights_in');
[weights, trainerr] = cnntrain(layers, params, train_x, train_y, funtype, weights_in);
plot(trainerr);
%save(fullfile(kWorkspaceFolder, 'weights.mat'), 'weights');
%%
%funtype = 'mexfun';
%funtype = 'matlab';
%load(fullfile(kWorkspaceFolder, 'weights.mat'), 'weights');
[pred, err, ~]  = cnntest(layers, weights, test_x, test_y, funtype);
disp([num2str(err*100) '% error']);
