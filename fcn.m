
close all; clear mex;

restoredefaultpath;
kCNNFolder = '.';  
kMexFolder = fullfile(kCNNFolder, 'c++', 'build');
addpath(kMexFolder);
addpath('./matlab');

addpath('/path/to/pascal-fcn32s-dag');
load pascal-fcn32s-dag;
PascalLayers = layers;
PascalWeights = params;
MeanX = single(meta.normalization.averageImage);

ClassesNum = 21;

kImagePath = './data/pascal';
ImList = dir(fullfile(kImagePath, '*.jpg'));
ImNum = numel(ImList);
JaccardIndices = -1 * ones(ImNum, ClassesNum);
confusion = zeros(ClassesNum);

for im = 1 : numel(ImList)

disp(im);
imname = ImList(im).name;
labname = strcat(imname(1:end-3), 'png');
if (~exist(fullfile(kImagePath, labname), 'file'))
  continue;
end;

TestX = imread(fullfile(kImagePath, imname));
TestY = imread(fullfile(kImagePath, labname));
test_x = TestX;

kXSize = size(TestX);
kXSize(end+1:4) = 1;

test_x = single(test_x) - repmat(MeanX, [kXSize(1) kXSize(2) 1]);

TestY = single(TestY);
TestY = mod(TestY + 1, 256);
test_y = single(zeros(kXSize(1), kXSize(2), ClassesNum));
for c = 1 : ClassesNum
  test_y(:, :, c) = (TestY == c);
end;

clear params;
params.epochs = 1;
params.alpha = 0.1;
params.beta = 0;
params.momentum = 0;
params.lossfun = 'logreg';
params.shuffle = 0;
params.seed = 1;
dropout = 0;
params.balance = 0;
params.verbose = 0;
params.gpu = 1;

layers = {
  struct('type', 'input', 'mapsize', kXSize(1:2), 'channels', kXSize(3))    
  % group 1
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 64, 'padding', [100 100])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 64, 'padding', [1 1])
  struct('type', 'pool', 'scale', [2 2], 'stride', [2 2])    
  % group 2
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 128, 'padding', [1 1])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 128, 'padding', [1 1])
  struct('type', 'pool', 'scale', [2 2], 'stride', [2 2])
  % group 3
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 256, 'padding', [1 1])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 256, 'padding', [1 1])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 256, 'padding', [1 1])
  struct('type', 'pool', 'scale', [2 2], 'stride', [2 2])
  % group 4
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 512, 'padding', [1 1])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 512, 'padding', [1 1])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 512, 'padding', [1 1])  
  struct('type', 'pool', 'scale', [2 2], 'stride', [2 2])
  % group 5
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 512, 'padding', [1 1])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 512, 'padding', [1 1])
  struct('type', 'conv', 'filtersize', [3 3], 'channels', 512, 'padding', [1 1])  
  struct('type', 'pool', 'scale', [2 2], 'stride', [2 2])  
  % group 6
  struct('type', 'conv', 'filtersize', [7 7], 'channels', 4096)  
  % group 7
  struct('type', 'conv', 'filtersize', [1 1], 'channels', 4096)
  % score2
  struct('type', 'conv', 'filtersize', [1 1], 'channels', ClassesNum, ...
         'function', 'none')
  % upscore_s
  struct('type', 'deconv', 'filtersize', [64 64], 'channels', ClassesNum, ...
         'stride', [32 32], 'function', 'none')
  % crop with softmax
  struct('type', 'jitt', 'mapsize', kXSize(1:2), 'function', 'soft')
};

layers = setup(layers);
%layers = genweights(layers, params, 'matlab');
layers = import_weights(layers, PascalWeights, numel(layers));

EpochNum = 1;
errors = zeros(EpochNum, 1);

for i = 1 : EpochNum
  disp(['Epoch: ' num2str((i-1) * params.epochs + 1)]);
  
  [err, bad, pred_y] = test(layers, params, test_x, test_y);
  [~, pred_ind] = max(pred_y, [], 3);
  test_y = logical(test_y);
  
  ok = TestY > 0;
  confusion = confusion + accumarray([TestY(ok), pred_ind(ok)], 1, [21 21]);
  
  subplot(2,2,1);
  imshow(TestX);  
  subplot(2,2,2);
  imshow(test_y(:, :, 1));  
  bg = pred_y(:, :, 1);
  bg = (max(bg(:))-bg)/(max(bg(:))-min(bg(:)));  
  subplot(2,2,3);  
  imshow(bg);
  [~, pred_ind] = max(pred_y, [], 3);
  binary = (pred_ind(:, :, 1) == 2);  
  subplot(2,2,4);  
  imshow(binary);
  
end;
disp('Done!');

end;

[IU, meanIU, pixelAccuracy, meanAccuracy] = get_accuracies(confusion)