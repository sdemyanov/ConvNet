kDatasetFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/SVHN/data';

load(fullfile(kDatasetFolder, 'train_32x32.mat'));
samplesNum = size(X, 4);

xSize = [size(X, 1) size(X, 2) size(X, 3)];
classesNum = 10;

%TrainX = single(X) / 255;
TrainX = X;
TrainY = single(zeros(samplesNum, classesNum));
for j = 1 : classesNum
  TrainY(y == j, j) = 1;
end;

validNum = 6000;
ValidX = uint8(zeros([xSize validNum]));
ValidY = single(zeros([validNum classesNum]));
curoff = 0;
classoff = 400;
for j = 1 : classesNum
  curind = find(TrainY(:, j) == 1);
  curind = curind(1:classoff);
  ValidX(:, :, :, curoff + 1: curoff + classoff) = TrainX(:, :, :, curind);
  ValidY(curoff + 1: curoff + classoff, :) = TrainY(curind, :);
  TrainX(:, :, :, curind) = [];  
  TrainY(curind, :) = [];
  curoff = curoff + classoff;
end;

load(fullfile(kDatasetFolder, 'extra_32x32.mat'));
samplesNum = size(X, 4);

%ExtraX = single(X) / 255;
ExtraX = X;
ExtraY = single(zeros(samplesNum, classesNum));
for j = 1 : classesNum
  ExtraY(y == j, j) = 1;
end;

classoff = 200;
for j = 1 : classesNum
  curind = find(ExtraY(:, j) == 1);
  curind = curind(1:classoff);
  ValidX(:, :, :, curoff + 1: curoff + classoff) = ExtraX(:, :, :, curind);
  ValidY(curoff + 1: curoff + classoff, :) = ExtraY(curind, :);
  ExtraX(:, :, :, curind) = [];  
  ExtraY(curind, :) = [];
  curoff = curoff + classoff;  
end;

TrainX(:, :, :, end+1 : end+size(ExtraX, 4)) = ExtraX;
TrainY(end+1 : end+size(ExtraY, 1), :) = ExtraY;

load(fullfile(kDatasetFolder, 'test_32x32.mat'));
samplesNum = size(X, 4);

%TestX = single(X) / 255;
TestX = X;
TestY = single(zeros(samplesNum, classesNum));
for j = 1 : classesNum
  TestY(y == j, j) = 1;
end;

save(fullfile(kDatasetFolder, 'svhn.mat'), 'TrainX', 'TrainY', 'TestX', 'TestY', 'ValidX', 'ValidY');


