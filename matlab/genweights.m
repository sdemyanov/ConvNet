function weights = genweights(layers, type)

if strcmp(type, 'mexfun')
  weights = genweights_mex(layers);
elseif strcmp(type, 'matlab')
  weights = genweights_mat(layers);
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;

end

