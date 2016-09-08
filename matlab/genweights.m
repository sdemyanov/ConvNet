function layers = genweights(layers, params, type)

if (strcmp(type, 'gpu'))
  weights = genweights_mex(layers, params);  
  layers = setweights(layers, weights);
elseif strcmp(type, 'matlab')
  layers = genweights_mat(layers, params);
else
  error('"%s" - wrong type, must be either "gpu" or "matlab"', type);
end;

end