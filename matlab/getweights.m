function weights = getweights(layers)

n = numel(layers);
ind = 0;
weights = single([]);
for l = 1 : n   %  layer
  
  if (isfield(layers{l}, 'w'))
    curlen = numel(layers{l}.w);
    weights(ind+1:ind+curlen, 1) = layers{l}.w(:);
    ind = ind + curlen;
  end;
  
  if (isfield(layers{l}, 'b'))
    curlen = numel(layers{l}.b);
    weights(ind+1:ind+curlen, 1) = layers{l}.b(:);
    ind = ind + curlen;
  end;
  
end

end