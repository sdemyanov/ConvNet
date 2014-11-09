function [weights, trainerr] = cnntrain(layers, weights_in, params, train_x, train_y, type)

if (length(size(train_x)) == 3)
  % insert singletone maps index
  train_x = permute(train_x, [1 2 4 3]); 
end;

tic;
if (strcmp(type, 'cpu') || strcmp(type, 'gpu'))
  [weights, trainerr] = cnntrain_mex(layers, weights_in, params, train_x, train_y);      
elseif strcmp(type, 'matlab')
  [weights, trainerr] = cnntrain_mat(layers, weights_in, params, train_x, train_y);
else
  error('"%s" - wrong type, must be either "cpu", "gpu" or "matlab"', type);
end;
t = toc;
disp(['Total training time: ' num2str(t)]);

end
