function weights = getweights(layers)

n = numel(layers);
ind = 0;
weights = double([]);
for l = 1 : n   %  layer  
  if strcmp(layers{l}.type, 'c') % convolutional
    k_trans = permute(layers{l}.k, [2 1 3 4]);
    curlen = length(k_trans(:));
    weights(ind+1:ind+curlen, 1) = k_trans(:);
    ind = ind + curlen;     
    curlen = length(layers{l}.b);
    weights(ind+1:ind+curlen, 1) = layers{l}.b;
    ind = ind + curlen;
  elseif strcmp(layers{l}.type, 'f') % fully connected
    curlen = numel(layers{l}.w);
    w_trans = layers{l}.w';
    weights(ind+1:ind+curlen, 1) = w_trans(:);
    ind = ind + curlen;
    curlen = length(layers{l}.b);
    weights(ind+1:ind+curlen, 1) = layers{l}.b;
    ind = ind + curlen;
  end;
end

end