function [weights, trainerr] = cnntrain_inv(layers, weights_in, train_x, train_y, params, type)

if(~iscell(train_x))
  if (length(size(train_x)) <= 3)
    % insert singletone maps index
    train_x = permute(train_x, [1 2 4 3]); 
  end;
else
  for i = 1 : numel(train_x)
    if (length(size(train_x{i})) <= 3)
      train_x{i} = permute(train_x{i}, [1 2 4 3]); 
    end;
  end;
end;

tic;
if strcmp(type, 'mexfun')
  ind = reshape(1 : prod(layers{2}{1}.mapsize), layers{2}{1}.mapsize)';
  if(~iscell(train_x))
    train_x = train_x(:, :, :, ind(:));    
    train_x = permute(train_x, [2 1 3 4]);
  else
    for i = 1 : numel(train_x)
      train_x{i} = train_x{i}(:, :, :, ind(:));
      train_x{i} = permute(train_x{i}, [2 1 3 4]);
    end;
  end;
  train_y = train_y';
  for i = 1 : length(layers)
    if (isfield(layers{i}{1}, 'mean'))
      layers{i}{1}.mean = permute(layers{i}{1}.mean, [2 1 3]);
    end;
    if (isfield(layers{i}{1}, 'stdev'))
      layers{i}{1}.stdev = permute(layers{i}{1}.stdev, [2 1 3]);
    end;
  end;
  cellweights = cell(size(weights_in{1}, 2), 1);
  for i = 1 : size(weights_in{1}, 2)
    cellweights{i} = weights_in{1}(:, i);
  end;
  weights_in{1} = cellweights;  
  [weights, trainerr] = cnntrain_inv_mex(layers, weights_in, train_x, train_y, params);  
  arrweights = zeros(length(weights{1}{1}), length(weights{1}));
  for i = 1 : size(arrweights, 2)
    arrweights(:, i) = weights{1}{i};    
  end;
  weights{1} = arrweights;  
  cellerr = trainerr;
  trainerr = zeros(size(cellerr{1}, 1), size(cellerr{1}, 2), 2);
  trainerr(:, :, 1) = cellerr{1};
  trainerr(:, :, 2) = cellerr{2};
elseif strcmp(type, 'matlab')
  [weights, trainerr] = cnntrain_inv_mat(layers, weights_in, train_x, train_y, params);
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;
t = toc;
disp(['Total training time: ' num2str(t)]);

st = size(trainerr);
trainerr = reshape(trainerr, st(1) * st(2), size(trainerr, 3));

end
