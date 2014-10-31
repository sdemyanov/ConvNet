function weights = genweights(layers, seed, type)

if (strcmp(type, 'cpu') || strcmp(type, 'gpu'))
  weights = genweights_mex(layers, seed);  
elseif strcmp(type, 'matlab')
  weights = genweights_mat(layers, seed);
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;

end

