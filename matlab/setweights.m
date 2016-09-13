function layers = setweights(layers, weights)

layers = setup(layers);
n = numel(layers);
ind = 0;
for l = 1 : n   %  layer  
  
  if (isfield(layers{l}, 'weightsize'))
    curlen = prod(layers{l}.weightsize);
    layers{l}.w = weights(ind+1:ind+curlen);
    layers{l}.w = reshape(layers{l}.w, layers{l}.weightsize);
    ind = ind + curlen;
  end;
  
  if (isfield(layers{l}, 'biassize'))
    curlen = prod(layers{l}.biassize);
    layers{l}.b = weights(ind+1:ind+curlen);
    layers{l}.b = reshape(layers{l}.b, layers{l}.biassize);
    ind = ind + curlen;
  end;
  
end
assert(ind == size(weights, 1), 'Weights vector is too long!');

end