imagesize = [32 32 3];
batchsize = 10000;
batchnum = 5;
classnum = 10;
kTrainNum = batchsize * batchnum;
kTestnum = batchsize;
TrainX = zeros([imagesize kTrainNum]);
TrainY = zeros(kTrainNum, classnum);
for i = 1 : batchnum
  load(['data_batch_' num2str(i) '.mat']);  
  curdata = double(data)' / 255;
  curind = batchsize * (i - 1) + 1 : batchsize * i;
  TrainX(:, :, :, curind) = reshape(curdata, [imagesize batchsize]);
  curlabels = zeros(batchsize, classnum);
  for j = 1 : classnum
    curlabels(labels == j - 1, j) = 1;
  end;
  TrainY(curind, :) = double(curlabels);
end;
TrainX = permute(TrainX, [2 1 3 4]);

load('test_batch.mat');
curdata = double(data)' / 255;
TestX = reshape(curdata, [imagesize batchsize]);
curlabels = zeros(batchsize, classnum);
for j = 1 : classnum
  curlabels(labels == j - 1, j) = 1;
end;
TestY = double(curlabels);
TestX = permute(TestX, [2 1 3 4]);

save('cifar.mat', 'TrainX', 'TrainY', 'TestX', 'TestY');
  
  
  