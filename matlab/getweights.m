function der = getweights(layers)

n = numel(layers);
ind = 0;
der = double([]);
for l = 1 : n   %  layer  
  if strcmp(layers{l}.type, 'i') % input
    if (isfield(layers{l}, 'mean'))      
      curlen = numel(layers{l}.mw);
      w_trans = permute(layers{l}.mw, [2 1 3 4]);
      der(ind+1:ind+curlen, 1) = w_trans(:);
      ind = ind + curlen;
    end;
    if (isfield(layers{l}, 'maxdev'))      
      curlen = numel(layers{l}.sw);
      w_trans = permute(layers{l}.sw, [2 1 3 4]);
      der(ind+1:ind+curlen, 1) = w_trans(:);
      ind = ind + curlen;
    end;
  elseif strcmp(layers{l}.type, 'n') % normalization
    curlen = numel(layers{l}.w);
    w_trans = permute(layers{l}.w, [2 1 3 4]);
    der(ind+1:ind+curlen, 1) = w_trans(:);
    ind = ind + curlen;     
  elseif strcmp(layers{l}.type, 'c') % convolutional
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

end