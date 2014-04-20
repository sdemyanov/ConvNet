function [weights, trainerr] = cnntrain_mat(layers, weights, train_x, train_y, params)

params = setparams(params);

layers = cnnsetup(layers, 0);
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
layers = setweights(layers, weights);

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
    layers = updateweights(layers, params, i, 0); % preliminary update
    layers = initact(layers, batch_x);
    [layers, pred] = forward(layers, 1);
    [layers, loss] = initder(layers, batch_y);
    layers = backward(layers);
    layers = calcweights(layers);
    layers = updateweights(layers, params, i, 1); % final update
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
