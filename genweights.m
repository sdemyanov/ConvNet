function weights = genweights(layers, seed, type)

if strcmp(type, 'mexfun')
  for i = 1 : numel(layers)
    if (isfield(layers{i}, 'mean'))
      layers{i}.mean = permute(layers{i}.mean, [2 1 3]);
    end;
    if (isfield(layers{i}, 'stdev'))
      layers{i}.stdev = permute(layers{i}.stdev, [2 1 3]);
    end;  
  end;
  weights = genweights_mex(layers, seed);
elseif strcmp(type, 'matlab')
  weights = genweights_mat(layers, seed);
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;

end

