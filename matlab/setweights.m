function layers = setweights(layers, weights)

n = numel(layers);
ind = 0;
for l = 2 : n   %  layer  
  if strcmp(layers{l}.type, 'c') % convolutional
    curlen = prod(layers{l}.kernelsize);
    for i = 1 : layers{l}.outputmaps  %  output map
      for j = 1 : layers{l-1}.outputmaps  %  input map
        k_trans = layers{l}.k{i, j}';
        k_trans(:) = weights(ind+1:ind+curlen);
        layers{l}.k{i, j} = k_trans';
        ind = ind + curlen;
      end      
    end
    curlen = layers{l}.outputmaps;
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
%layers{end}.coef = weights(ind+1:ind+layers{end}.length);
%ind = ind + layers{end}.length;

end