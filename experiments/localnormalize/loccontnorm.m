
if (ispc)
  kDatasetFolder = 'C:/Users/sergeyd/Workspaces/SVHN/data';  
else
  kDatasetFolder = '/media/sergeyd/OS/Users/sergeyd/Workspaces/SVHN/data';  
end;

%load(fullfile(kDatasetFolder, 'svhn.mat'), 'TrainX');  
%ds = TrainX;
%load(fullfile(kDatasetFolder, 'svhn.mat'), 'TestX');
%ds = TestX;
t = 3;
if (t == 1)
  load(fullfile(kDatasetFolder, 'svhn.mat'), 'TrainX', 'TrainY');
  ds = TrainX; lb = TrainY;
  clear TrainX;
elseif (t == 2)
  load(fullfile(kDatasetFolder, 'svhn.mat'), 'ValidX', 'ValidY');
  ds = ValidX; lb = ValidY;
  clear ValidX;
elseif (t == 3)
  load(fullfile(kDatasetFolder, 'svhn.mat'), 'TestX', 'TestY');
  ds = TestX; lb = TestY;
  clear TestX;
end;
xSize = [size(ds, 1) size(ds, 2) size(ds, 3)];
pixNum = prod(xSize);
sampleNum = size(ds, 4);

dsnew = single(zeros(size(ds)));

batchsize = 10000;
batchnum = sampleNum / batchsize + 1;

for bind = 1 : batchnum
  curind = (bind-1) * batchsize + 1 : min(bind*batchsize, sampleNum);
  cursize = length(curind);
  batch = ds(:, :, :, curind);
  batch = single(batch) / 255;
  batch = reshape(batch, pixNum, cursize);
  batch_y = lb(curind, :);

  % global contrast normalization
  samplemean = mean(batch, 1);
  batch = batch - repmat(samplemean, [pixNum 1]);
  samplenorm = sqrt(sum(batch.^2, 1));
  batch = batch ./ repmat(samplenorm, pixNum, 1);
  batch = reshape(batch, [xSize cursize]);
  
  % local contrast normalization
  hsize = [3 3];
  fsize = 2 * hsize + 1;  
  sigma = 2;
  fm = fspecial('gaussian', fsize, sigma);
  %fm = fspecial('average', fsize);
  locmean = convn(batch, fm, 'same');
  batch = batch - locmean;  

  ds_sum_sq = sqrt(convn(batch.^2, fm, 'same'));
  per_img_mean = mean(mean(ds_sum_sq, 2), 1);
  divisor = max(repmat(per_img_mean, [xSize(1) xSize(2) 1 1]), ds_sum_sq);
  kEps = 1e-8;
  disp(length(find(divisor < kEps)));
  divisor(divisor < kEps) = kEps;

  batch = batch ./ divisor;
  %dsnew(:, :, :, curind) = batch;
  
  batch_x = batch;
  if (t == 1)
    batch_name = ['train-batch-' num2str(bind) '.mat'];
  elseif (t == 2)
    batch_name = ['valid-batch-' num2str(bind) '.mat'];
  elseif (t == 3)
    batch_name = ['test-batch-' num2str(bind) '.mat'];
  end;  
  save(fullfile(kDatasetFolder, batch_name), 'batch_x', 'batch_y');

end;
%{
if (t == 1)
  TrainX = dsnew;
  save(fullfile(kDatasetFolder, 'svhn-norm.mat'), 'TrainX', '-append');  
elseif (t == 2)
  ValidX = dsnew;
  save(fullfile(kDatasetFolder, 'svhn-norm.mat'), 'ValidX', '-append');  
elseif (t == 3)
  TestX = dsnew;
  save(fullfile(kDatasetFolder, 'svhn-norm.mat'), 'TestX', '-append');
end;
%}