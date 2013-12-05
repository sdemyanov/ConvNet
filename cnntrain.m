function [weights, trainerr] = cnntrain(varargin)

layers = varargin{1};
params = varargin{2};
train_x = varargin{3};
train_y = varargin{4};
type = varargin{5};
if (nargin > 5)
  weights = varargin{6};
end;

tic;
if strcmp(type, 'mexfun')
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

end
