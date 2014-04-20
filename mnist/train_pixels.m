close all; clear mex;

kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\data';
load(fullfile(kWorkspaceFolder, 'mnist.mat'));
%TrainX = double(permute(reshape(train_x', [kXSize size(train_x, 1)]), [2 1 3]))/255;

kCNNFolder = '.';
addpath(fullfile(kCNNFolder, 'c++/build/'));
addpath(fullfile(kCNNFolder, 'matlab'));

kXSize = [28 28];
%kTrainNum = size(TestX, 3);
kStartIter = 7501;
kTrainNum = 2500;
kOutputs = 2;

layers = {
  struct('type', 'i', 'mapsize', [1 2], 'outputmaps', 1)
  %struct('type', 'c', 'kernelsize', [2 2], 'outputmaps', 5, 'function', 'relu', 'padding', [1 1]);
  %struct('type', 'c', 'kernelsize', [2 2], 'outputmaps', 5, 'function', 'relu');
  struct('type', 'f', 'length', 8, 'function', 'relu')
  struct('type', 'f', 'length', 16, 'function', 'relu')
  struct('type', 'f', 'length', 8, 'function', 'relu')
  %struct('type', 'f', 'length', 30, 'function', 'relu')    
  %struct('type', 'f', 'length', 20, 'function', 'relu')    
  struct('type', 'f', 'length', 1, 'function', 'sigm')
};

%weights_in_pixels = genweights(layers, funtype);
%save(fullfile(kWorkspaceFolder, 'weights_in_pixels.mat'), 'weights_in_pixels');

params.alpha = 1;
params.batchsize = 50;
params.numepochs = 3000;
params.momentum = 0;  
params.adjustrate = 0;
params.maxcoef = 10;
params.balance = 0;
params.shuffle = 1;
params.verbose = 0;

Threshold = 0.15;
MaxIterNum = 15;

s = kXSize;
train_x = zeros(s(1) * s(2), 2);
ind1 = repmat((1:s(1))', [1 s(2)]) / s(1) - 0.5;
train_x(:, 1) = ind1(:);
ind2 = repmat(1:s(2), [s(1) 1]) / s(2) - 0.5;
train_x(:, 2) = ind2(:);
train_x = permute(train_x, [3 2 1]);  
funtype = 'mexfun';
weights_in = genweights(layers, funtype);
Weights = zeros(kTrainNum, length(weights_in));
Errors = zeros(kTrainNum, 1);
%figure;
for iter = kStartIter : kStartIter + kTrainNum - 1
  %im = imread(fullfile(kSourceFolder, strcat('cat', '.', num2str(iter-1), '.jpg')));
  %im = cv.cvtColor(im,'RGB2GRAY');
  %im = double(im)/255;
  im = TestX(:,:,iter);  
  %subplot(1,2,1); imshow(im); drawnow;  
  %norm_x = sum(sum(im.^2)) / (s(1) * s(2));
  train_y = im(:);
  
  err = 10; minerr = err; iternum = 0;
  while (err > Threshold && iternum <= MaxIterNum)
    %funtype = 'matlab';
    %load(fullfile(kWorkspaceFolder, 'weights_in_pixels.mat'), 'weights_in_pixels');
    weights = genweights(layers, funtype);
    [weights, trainerr] = cnntrain(layers, weights, train_x, train_y, params, funtype);
    %subplot(1,3,3); plot(trainerr);    
    %save(fullfile(kWorkspaceFolder, 'weights.mat'), 'weights');
    %%
    %funtype = 'mexfun';
    %funtype = 'matlab';
    %load(fullfile(kWorkspaceFolder, 'weights.mat'), 'weights');
    pred = cnnclassify(layers, weights, train_x, funtype);
    pred = reshape(pred, s);
    %subplot(1,2,2); imshow(pred); drawnow;
    %norm_p = sum(sum(pred.^2)) / (s(1) * s(2));
    %pred = pred * norm_x / norm_p;
    err = sum(abs(pred(:) - train_y)) / sum(train_y);
    if (err < minerr)
      minerr = err;
      minweights = weights;
    end;
    disp([num2str(err * 100) '% error']);
    iternum = iternum + 1;
  end;
  Weights(iter, :) = minweights;
  Errors(iter) = minerr;
  %figure; plot(err); drawnow();
  disp(iter);
end;

finalfile = ['weights-test' num2str(kStartIter) '.mat'];
save(fullfile(kWorkspaceFolder, finalfile), 'Weights', 'Errors');
