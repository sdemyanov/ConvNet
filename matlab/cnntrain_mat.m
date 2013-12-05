function [weights, trainerr] = cnntrain_mat(varargin)

layers = varargin{1};
params = varargin{2};
train_x = varargin{3};
train_y = varargin{4};
if (nargin > 4)
  weights = varargin{5};
end;

layers = cnnsetup(layers);
if (nargin > 4)
  layers = setweights(layers, weights);
end;
params = setparams(params);
if (~iscell(train_x))
  copy_x = train_x;
  clear train_x;
  train_x{1} = copy_x;
end;
train_num = size(train_x{1}, 3);
assert(length(train_x) == layers{1}.outputmaps, ...
  'Data must have the same number of cells as outputmaps on the first layer');  
for k = 1:length(train_x)
  assert(length(size(train_x{k})) == 3, 'The data array must have 3 dimensions');  
  assert(size(train_x{k}, 1) == layers{1}.mapsize(1) && ...
       size(train_x{k}, 2) == layers{1}.mapsize(2), ...
       'Data and the first layer must have equal sizes');
end;
assert(length(size(train_y)) == 2, 'The label array must have 2 dimensions'); 
assert(size(train_y, 2) == train_num, 'Data and labels must have equal number of objects');
assert(size(train_y, 1) == layers{end}.length, 'Labels and last layer must have equal number of classes');
if (params.balance == 1)
  layers{end}.coef = mean(train_y, 2);
elseif (params.balance == 0)
  layers{end}.coef = ones(size(train_y, 1), 1) / 2;  
end;

numbatches = ceil(train_num/params.batchsize);
trainerr = zeros(params.numepochs * numbatches, 1);
for i = 1 : params.numepochs
  if (nargin == 5)
    kk = 1:train_num;
  else
    kk = randperm(train_num);
  end;
  for j = 1 : numbatches
    batch_x = cell(length(train_x));
    for k = 1:length(train_x)
      batch_x{k} = train_x{k}(:, :, kk((j-1)*params.batchsize + 1 : min(j*params.batchsize, train_num)));
    end;
    batch_y = train_y(:, kk((j-1)*params.batchsize + 1 : min(j*params.batchsize, train_num)));
    tic;
    layers = updateweights(layers, params, 0); % preliminary update
    [layers, ~] = cnnff(layers, batch_x, 1);
    [layers, loss] = cnnbp(layers, batch_y);
    layers = updateweights(layers, params, 1); % final update
    trainerr((i-1)*numbatches+j) = loss;    
    disp([i j]);
  end  
end
    
weights = getweights(layers);

end
