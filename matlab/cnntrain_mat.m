function [weights, trainerr] = cnntrain_mat(varargin)

layers = varargin{1};
params = varargin{2};
train_x = varargin{3};
train_y = varargin{4};
if (nargin > 4)
  weights = varargin{5};
end;

layers = cnnsetup(layers);

assert(length(size(train_y)) == 2, 'The label array must have 2 dimensions'); 
train_num = size(train_y, 1);
classes_num = size(train_y, 2);
assert(classes_num == layers{end}.length, 'Labels and last layer must have equal number of classes');
if (params.balance == 1)
  layers{end}.coef = ones(1, classes_num) ./ mean(train_y, 1) / classes_num;
elseif (params.balance == 0)
  layers{end}.coef = ones(1, classes_num);  
end;
if strcmp(layers{end}.function, 'SVM')
  train_y(train_y == 0) = -1;
end;

if (nargin > 4)
  layers = setweights(layers, weights);
end;
params = setparams(params);
assert(length(size(train_x)) == 3 || length(size(train_x)) == 4, ...
       'Wrong dimensionality of input data');
assert(size(train_x, 1) == layers{1}.mapsize(1) && ...
       size(train_x, 2) == layers{1}.mapsize(2), ...
       'Data and the first layer must have equal sizes');
assert(size(train_x, 3) == layers{1}.outputmaps, ...
       'The number of the input data maps must be as specified');
assert(size(train_x, 4) == train_num, ...
       'Data and labels must have equal number of objects');


numbatches = ceil(train_num/params.batchsize);
trainerr = zeros(params.numepochs, numbatches);
for i = 1 : params.numepochs
  if (params.shuffle == 0)
    kk = 1:train_num;
  else
    kk = randperm(train_num);
  end;
  for j = 1 : numbatches
    batch_x = train_x(:, :, :, kk((j-1)*params.batchsize + 1 : min(j*params.batchsize, train_num)));    
    batch_y = train_y(kk((j-1)*params.batchsize + 1 : min(j*params.batchsize, train_num)), :);
    tic;
    layers = updateweights(layers, params, 0); % preliminary update
    [layers, ~] = cnnff(layers, batch_x, 1);
    [layers, loss] = cnnbp(layers, batch_y);
    layers = updateweights(layers, params, 1); % final update
    trainerr(i, j) = loss;    
    if (params.verbose == 2)
      disp(['Epoch: ' num2str(i) ', batch: ', num2str(j)]);
    end;
  end
  if (params.verbose == 1)
    disp(['Epoch: ' num2str(i)]);
  end;
end
    
weights = getweights(layers);
trainerr = trainerr';

end
