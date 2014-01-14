function der = getweights(layers)

n = numel(layers);
ind = 0;
der = double([]);
for l = 2 : n   %  layer  
  if strcmp(layers{l}.type, 'c') % convolutional
    k_trans = permute(layers{l}.k, [2 1 3 4]);
    curlen = length(k_trans(:));
    der(ind+1:ind+curlen, 1) = k_trans(:);
    ind = ind + curlen;     
    curlen = numel(layers{l}.b);
    der(ind+1:ind+curlen, 1) = layers{l}.b;
    ind = ind + curlen;
  elseif strcmp(layers{l}.type, 'f') % fully connected
    curlen = numel(layers{l}.w);
    w_trans = layers{l}.w';
    der(ind+1:ind+curlen, 1) = w_trans(:);
    ind = ind + curlen;
    curlen = numel(layers{l}.b);
    der(ind+1:ind+curlen, 1) = layers{l}.b;
    ind = ind + curlen;
  end;
end
%der(ind+1:ind+layers{end}.length) = layers{end}.coef;
%ind = ind + layers{end}.length;
%der(ind+1:end) = [];

end