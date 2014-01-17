function weights = genweights(layers, type)

if strcmp(type, 'mexfun')
  if (isfield(layers{1}, 'mean'))
    layers{1}.mean = permute(layers{1}.mean, [2 1 3]);
  end;
  if (isfield(layers{1}, 'stdev'))
    layers{1}.stdev = permute(layers{1}.stdev, [2 1 3]);
  end;  
  weights = genweights_mex(layers);
elseif strcmp(type, 'matlab')
  weights = genweights_mat(layers);
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;

end

