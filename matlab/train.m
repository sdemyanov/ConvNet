function [weights, losses] = train(layers, weights, params, train_x, train_y)

tic;
% to get NCHW layout and float precision
train_x = single(permute(train_x, [2 1 3 4]));
train_y = single(permute(train_y, [2 1 3 4]));
%weights = getweights(layers);
[weights, losses] = train_mex(layers, weights, params, train_x, train_y);
%layers = setweights(layers, weights);
t = toc;

disp(['Total training time: ' num2str(t)]);

end
