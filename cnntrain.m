function [weights, trainerr] = cnntrain(varargin)

layers = varargin{1};
params = varargin{2};
train_x = varargin{3};
train_y = varargin{4};
type = varargin{5};
if (nargin > 5)
  weights = varargin{6};
end;

if (length(size(train_x)) == 3)
  % insert singletone maps index
  train_x = permute(train_x, [1 2 4 3]); 
end;

tic;
if strcmp(type, 'mexfun')
  train_x = permute(train_x, [2 1 3 4]);
  train_y = train_y';
  if (isfield(layers{1}, 'mean'))
    layers{1}.mean = permute(layers{1}.mean, [2 1 3]);
  end;
  if (isfield(layers{1}, 'stdev'))
    layers{1}.stdev = permute(layers{1}.stdev, [2 1 3]);
  end;
  if (nargin > 5)
    [weights, trainerr] = cnntrain_mex(layers, params, train_x, train_y, weights);
  else
    [weights, trainerr] = cnntrain_mex(layers, params, train_x, train_y);
  end;
elseif strcmp(type, 'matlab')
  if (nargin > 5)
    [weights, trainerr] = cnntrain_mat(layers, params, train_x, train_y, weights);
  else
    [weights, trainerr] = cnntrain_mat(layers, params, train_x, train_y);
  end;  
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;
t = toc;
disp(['Total training time: ' num2str(t)]);

trainerr = trainerr(:);

end
