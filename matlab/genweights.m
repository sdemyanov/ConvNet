function weights = genweights(layers, params, type)

if (strcmp(type, 'gpu'))
  weights = genweights_mex(layers, params);  
elseif strcmp(type, 'matlab')
  weights = genweights_mat(layers, params);
else
  error('"%s" - wrong type, must be either "gpu" or "matlab"', type);
end;

end