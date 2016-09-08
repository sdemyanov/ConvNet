function layers = setweights(layers, weights)

n = numel(layers);
ind = 0;
for l = 1 : n   %  layer  
  
  if (~isfield(layers{l}, 'w'))
    continue;
  end;
  
  curlen = numel(layers{l}.w);
  layers{l}.w(:) = weights(ind+1:ind+curlen);
  ind = ind + curlen;
  
  if (~isfield(layers{l}, 'add_bias') || layers{l}.add_bias)
    curlen = numel(layers{l}.b);
    layers{l}.b(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;
  end;
  
end
assert(ind == size(weights, 1), 'Weights vector is too long!');

end