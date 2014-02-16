function [layers, loss] = initder(layers, y)

n = numel(layers);
batchsize = size(y, 1); % number of examples in the minibatch  
if (strcmp(layers{n}.function, 'sigm') || strcmp(layers{n}.function, 'soft'))
  layers{n}.d = layers{n}.a - y;
  loss = 1/2 * sum(layers{n}.d(:).^2) / batchsize;
elseif (strcmp(layers{n}.function, 'SVM')) % for SVM 
  layers{n}.d = -2 * y .* max(1 - layers{n}.a .* y, 0);      
  loss = sum(sum(max(1 - layers{n}.a .* y, 0).^2)) / batchsize;
  % + 1/2 * sum(sum(last_layer.w * last_layer.w')) / last_layer.C - too long
else
  error('The last layer function must be either "sigm" or "SVM"');
end;
layers{n}.d(-layers{n}.eps < layers{n}.d & layers{n}.d < layers{n}.eps) = 0;

end