
kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\CIFAR-100\data';

load(fullfile(kWorkspaceFolder, 'train.mat'));
kTrainNum = size(data, 1);
imagesize = [32 32 3];

TrainX = double(data) / 255;
TrainX = reshape(TrainX', [imagesize kTrainNum]);
TrainX = permute(TrainX, [2 1 3 4]);

classnum = 100;
TrainY = zeros(kTrainNum, classnum);

for j = 1 : classnum
  TrainY(fine_labels == j - 1, j) = 1;  
end;

TrainX = single(TrainX);
TrainY = single(TrainY);

save(fullfile(kWorkspaceFolder, 'cifar-100.mat'), 'TrainX', 'TrainY');

load(fullfile(kWorkspaceFolder, 'test.mat'));

TestX = double(data) / 255;
TestX = reshape(TestX', [imagesize kTrainNum]);
TestX = permute(TestX, [2 1 3 4]);

classnum = 100;
TestY = zeros(kTrainNum, classnum);

for j = 1 : classnum
  TestY(fine_labels == j - 1, j) = 1;  
end;

TestX = single(TestX);
TestY = single(TestY);

save(fullfile(kWorkspaceFolder, 'cifar-100.mat'), 'TestX', 'TestY', '-append');

