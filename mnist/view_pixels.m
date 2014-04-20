close all; clear mex;
%clear;

kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST';

is_show = 1;
is_train = 1;
is_ext = 0;

if (is_show == 1)
  if (is_ext == 0)
    load(fullfile(kWorkspaceFolder, 'data', 'mnist.mat'), 'TrainX', 'TestX');
  else
    load(fullfile(kWorkspaceFolder, 'data', 'mnist_ext.mat'), 'TrainX', 'TestX');
  end;
  if (is_train == 1)
    ShowX = TrainX;
  else
    ShowX = TestX;
  end;
end;

if (is_train == 1)
  load(fullfile(kWorkspaceFolder, 'weights', 'allweights.mat'), 'TrainWeights', 'TrainY');
  Weights = TrainWeights;
else
  if (is_ext == 0)
    load(fullfile(kWorkspaceFolder, 'weights', 'mnist_res.mat'), 'TrainX', 'TrainY');
  else
    load(fullfile(kWorkspaceFolder, 'weights', 'mnist_ext_res.mat'), 'TrainX', 'TrainY');
  end;
  load(fullfile(kWorkspaceFolder, 'weights', 'allweights.mat'), 'TestWeights', 'TestY');
  Weights = TestWeights;
end;

kCNNFolder = '..';
addpath(kCNNFolder);
addpath(fullfile(kCNNFolder, 'c++/build/'));
addpath(fullfile(kCNNFolder, 'matlab'));

layers = {
  struct('type', 'i', 'mapsize', [1 2], 'outputmaps', 1)
  struct('type', 'f', 'length', 8)
  struct('type', 'f', 'length', 16)
  struct('type', 'f', 'length', 8)
  struct('type', 'f', 'length', 1, 'function', 'sigm')
};

kCurNum = size(Weights, 1);

if (is_ext == 0)
  kXSize = [28 28];
  s = kXSize;
  ind1 = repmat((1:s(1))', [1 s(2)]) / s(1) - 0.5;
  ind2 = repmat(1:s(2), [s(1) 1]) / s(2) - 0.5;
  CurX = zeros(s(1) * s(2), 2);
  CurX(:, 1) = ind1(:);
  CurX(:, 2) = ind2(:);
  CurX = permute(CurX, [3 2 1]);  
elseif (is_ext == 1)
elseif (is_ext == 2)
  kXSize = [24 24];  
  kShift = [2 2];
  CurX = GetMultiCoords(kXSize, kShift);
  CurX = repmat(CurX, kCurNum, 1);
  Weights = expand(Weights, [5 1]);
  if (is_train == 1)
    TrainY = expand(TrainY, [5 1]);
  end;
  kCurNum = 5 * kCurNum;
end;

funtype = 'mexfun';
%funtype = 'matlab';  

Images = zeros([kXSize kCurNum]);
%figure;
for iter = 1 : kCurNum
  if (is_show == 1) 
    im = ShowX(:, :, iter);  
    subplot(1,2,1); imshow(im); drawnow;    
    title('Original');
  end;
  weights = Weights(iter, :);
  if (is_ext ~= 2)
    pred = cnnclassify(layers, weights, CurX, funtype);
  else
    pred = cnnclassify(layers, weights, CurX{iter}, funtype);
  end;
  pred = reshape(pred, kXSize);
  Images(:, :, iter) = pred;
  if (is_show == 1) 
    subplot(1,2,2); imshow(pred); drawnow;  
    title('Restored');
  end;
  disp(iter);
end;

if (is_train == 1)
  TrainX = Images;
  if (is_ext == 0)
    %save(fullfile(kWorkspaceFolder, 'mnist_res.mat'), 'TrainX', 'TrainY');
  else
    %save(fullfile(kWorkspaceFolder, 'mnist_ext_res.mat'), 'TrainX', 'TrainY');
  end;
else
  TestX = Images;
  if (is_ext == 0)
    %save(fullfile(kWorkspaceFolder, 'mnist_res.mat'), 'TrainX', 'TrainY', 'TestX', 'TestY');
  else
    %save(fullfile(kWorkspaceFolder, 'mnist_ext_res.mat'), 'TrainX', 'TrainY', 'TestX', 'TestY');
  end;    
end;



