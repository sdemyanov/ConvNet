
close all; clear mex;

kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST';
load(fullfile(kWorkspaceFolder, 'data', 'mnist.mat'), 'TestX', 'TestY');
%%
kTestNum = 10000;
TestX = TestX(:, :, 1:kTestNum);
TestY = double(TestY(1:kTestNum, :));

kOutputs = size(TestY, 2);
kCNNFolder = '..';
addpath(fullfile(kCNNFolder, 'c++/build/'));
addpath(fullfile(kCNNFolder, 'matlab'));

kXSize = [size(TestX, 1) size(TestX, 2)];
%{
ceps = -1e-4;
%products = [0.012776454684099 -0.006664378731082]; % orthogonal
products = [-0.006664378731082 -0.012776454684099]; % parallel
products = products / sqrt(sum(products.^2));
TestX = TestX + ceps * repmat(products, [1 1 1 size(TestX, 4)]);
%}

load(fullfile(kWorkspaceFolder, 'data', 'stats.mat'), 'norm_x', 'mean_x', 'std_x');

layers = {
  %struct('type', 'i', 'mapsize', kXSize, 'outputmaps', 1, ...
  %       'norm', norm_x, 'mean', mean_x, 'stdev', std_x) 
  struct('type', 'i', 'mapsize', kXSize, 'outputmaps', 1)
  struct('type', 'f', 'length', 60)
  struct('type', 'f', 'length', 60)  
  struct('type', 'f', 'length', kOutputs, 'function', 'soft')
};

funtype = 'mexfun';
%funtype = 'matlab';

kMaxIter = 30;
kWeightsLength = 51370;

n = 1;
Beta{n} = 'b'; n = n + 1;
Beta{n} = '0'; n = n + 1;
Beta{n} = '0.1'; n = n + 1;
Beta{n} = '0.2'; n = n + 1;
Beta{n} = '0.3'; n = n + 1;
BetaNum = length(Beta);
Weights = zeros(kWeightsLength, kMaxIter, n);
for i = 1 : BetaNum
  weightsfile = ['weights-' Beta{i} '.mat'];    
  disp(['Beta: ' Beta{i}]);
  load(fullfile(kWorkspaceFolder, 'Results_ff1000', weightsfile), 'WeightsIn');
  Weights(:, :, i) = WeightsIn(:, 1:kMaxIter);  
end;

shiftfile = 'jittering.mat'; 
kStartIter = 1;
kMaxEpoch = 50;
errors = zeros(kMaxEpoch, BetaNum, kMaxIter);
for iter = kStartIter : kMaxIter
for epoch = 1 : kMaxEpoch
  disp(['Iter: ' num2str(iter)]);
  disp(['Epoch: ' num2str(epoch)]);  
  ceps = epoch * 1e-2;
  rng(iter);
  pix_shift = ceps * repmat(kXSize, kTestNum, 1) .* (rand(kTestNum, 2) - 0.5);
  test_x = zeros(size(TestX));
  for i = 1 : kTestNum
    T = affine2d([1 0 0; 0 1 0; pix_shift(i, :) 1]); % represents translation
    test_x(:, :, i) = imwarp(TestX(:, :, i), T, 'OutputView', imref2d(kXSize));    
    %subplot(1, 2, 1); imshow(TestX(:, :, i));
    %subplot(1, 2, 2); imshow(test_x(:, :, i));
  end;
  for n = 1 : BetaNum
    [err, bad, pred] = cnntest(layers, Weights(:, iter, n), test_x, TestY, funtype);
    errors(epoch, n, iter) = err;    
    disp(['Err: ' num2str(err)]);
  end;    
  save(fullfile(kWorkspaceFolder, shiftfile), 'errors');  
end;
end;
%%




%{
products = [0.012776454684099 -0.006664378731082]; % orthogonal
products = [-0.006664378731082 -0.012776454684099]; % parallel
products = products ./ repmat(sqrt(sum(products.^2, 2)), [kTestNum 1]);
kNumIter = 1;
tl = zeros(kNumIter, 1);
for i = 1 : kNumIter
TestX = TestX + ceps * repmat(products, [1 1 1 size(TestX, 4)]);
[err, loss, loss2] = cnntest_ext(layers, weights, TestX, TestY, funtype);
%disp(loss);
tl(i) = loss;
end;
disp(loss);
%%
%plot(tl);
%axis([0 kNumIter 0.73 0.76]);
%}


