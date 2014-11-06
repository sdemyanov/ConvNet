function [layers, loss] = initder(layers, params, y)

n = numel(layers);
assert(strcmp(layers{n}.function, 'sigm') || strcmp(layers{n}.function, 'soft'), ...
  'The last layer function must be either "soft" or "sigm"');
batchsize = size(y, 1); % number of examples in the minibatch  
if (strcmp(params.lossfun, 'logreg'))
  lossmat = layers{n}.a;
  lossmat(y == 0) = 1;
  lossmat(lossmat == 0) = layers{n}.eps;
  if (strcmp(layers{n}.function, 'soft'))
    layers{n}.d = layers{n}.a - y;
  else
    layers{n}.d = -y ./ lossmat;
  end;
  loss = -sum(log(lossmat(:))) / batchsize;
else
  layers{n}.d = layers{n}.a - y;
  loss = 1/2 * sum(layers{n}.d(:).^2) / batchsize;  
end;
layers{n}.d(-layers{n}.eps < layers{n}.d & layers{n}.d < layers{n}.eps) = 0;

end