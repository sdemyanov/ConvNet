function [layers, loss] = initder2(layers)

batchsize = size(layers{1}.d, 4);
deriv_sq = layers{1}.d.^2;
loss = sum(deriv_sq(:)) / (2 * batchsize);
layers{1}.a = layers{1}.d;

end
