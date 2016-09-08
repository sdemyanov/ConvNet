function weights = getweights(layers)

n = numel(layers);
ind = 0;
weights = single([]);
for l = 1 : n   %  layer
  
  if (~isfield(layers{l}, 'w'))
    continue;
  end;
  
  curlen = numel(layers{l}.w);
  weights(ind+1:ind+curlen, 1) = layers{l}.w(:);
  ind = ind + curlen; 
  
  if (~isfield(layers{l}, 'add_bias') || layers{l}.add_bias)
    curlen = numel(layers{l}.b);
    weights(ind+1:ind+curlen, 1) = layers{l}.b(:);
    ind = ind + curlen;
  end;
  
end

end