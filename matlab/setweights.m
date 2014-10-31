function layers = setweights(layers, weights)

n = numel(layers);
ind = 0;
for l = 1 : n   %  layer  
  if strcmp(layers{l}.type, 'i') % input
    if (isfield(layers{l}, 'mean'))   
      curlen = length(layers{l}.mw(:));
      %w_trans = permute(layers{l}.mw, [2 1 3 4]);      
      %w_trans(:) = weights(ind+1:ind+curlen);
      %layers{l}.mw = permute(w_trans, [2 1 3 4]);
      layers{l}.mw(:) = weights(ind+1:ind+curlen);
      ind = ind + curlen; 
    end;
    if (isfield(layers{l}, 'maxdev'))   
      curlen = length(layers{l}.sw(:));
      %w_trans = permute(layers{l}.sw, [2 1 3 4]);      
      %w_trans(:) = weights(ind+1:ind+curlen);
      %layers{l}.sw = permute(w_trans, [2 1 3 4]);
      layers{l}.sw(:) = weights(ind+1:ind+curlen);      
      ind = ind + curlen; 
    end;
  elseif strcmp(layers{l}.type, 'n') % normalization
    w_trans = permute(layers{l}.w, [2 1 3 4]);
    curlen = length(w_trans(:));
    w_trans(:) = weights(ind+1:ind+curlen);
    layers{l}.w = permute(w_trans, [2 1 3 4]);
    ind = ind + curlen;        
  elseif strcmp(layers{l}.type, 'c') % convolutional    
    curlen = length(layers{l}.k(:));
    k_trans = permute(layers{l}.k, [4 1 2 3]);    
    k_trans(:) = weights(ind+1:ind+curlen);
    layers{l}.k = permute(k_trans, [2 3 4 1]);
    %layers{l}.k(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;    
    curlen = layers{l}.outputmaps;
    layers{l}.b(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;
  elseif strcmp(layers{l}.type, 'f') % fully connected
    curlen = numel(layers{l}.w);
    %w_trans = layers{l}.w';
    %w_trans(:) = weights(ind+1:ind+curlen);
    %layers{l}.w = w_trans';
    layers{l}.w(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;
    curlen = numel(layers{l}.b);
    layers{l}.b(:) = weights(ind+1:ind+curlen);
    ind = ind + curlen;
  end;
end
assert(ind == size(weights, 1), 'Weights vector is too long!');

end