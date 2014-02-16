function layers = setweights(layers, weights)

n = numel(layers);
ind = 0;
for l = 1 : n   %  layer  
  if strcmp(layers{l}.type, 'c') % convolutional
    %if (layers{l}.common)
    k_trans = permute(layers{l}.k, [2 1 3 4]);
    curlen = length(k_trans(:));
    k_trans(:) = weights(ind+1:ind+curlen);
    layers{l}.k = permute(k_trans, [2 1 3 4]);
    ind = ind + curlen;    
    curlen = layers{l}.outputmaps;
    layers{l}.b(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;      
    %{
    else
      s = size(layers{l}.k);
      s(5:end) = []; s(end+1:4) = 1;
      k_trans = zeros([s size(weights, 2)]);
      curlen = prod(s);
      k_trans(:) = weights(ind+1:ind+curlen, :);
      layers{l}.k = permute(k_trans, [2 1 3 4 5]);
      ind = ind + curlen;    
      curlen = layers{l}.outputmaps;
      layers{l}.b = weights(ind+1:ind+curlen, :);
      ind = ind + curlen;      
    end;
    %}
    
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
assert(ind == size(weights, 1), 'Weights vector is too long!');

end