function layers = setweights(layers, weights)

n = numel(layers);
ind = 0;
for l = 1 : n   %  layer  
  if strcmp(layers{l}.type, 'c') % convolutional
    k_trans = permute(layers{l}.k, [2 1 3 4]);
    curlen = length(k_trans(:));
    k_trans(:) = weights(ind+1:ind+curlen);
    layers{l}.k = permute(k_trans, [2 1 3 4]);
    ind = ind + curlen;    
    curlen = length(layers{l}.b);
    layers{l}.b(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;
  elseif strcmp(layers{l}.type, 'f') % fully connected
    curlen = numel(layers{l}.w);
    w_trans = layers{l}.w';
    w_trans(:) = weights(ind+1:ind+curlen);
    layers{l}.w = w_trans';
    ind = ind + curlen;
    curlen = length(layers{l}.b);
    layers{l}.b(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;
  end;
end
assert(ind == length(weights), 'Weights vector is too long!');

end