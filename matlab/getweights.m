function der = getweights(layers)

n = numel(layers);
ind = 0;
der = zeros(10000, 1);
for l = 2 : n   %  layer  
  if strcmp(layers{l}.type, 'c') % convolutional
    curlen = prod(layers{l}.kernelsize);
    for i = 1 : layers{l}.outputmaps  %  output map
      for j = 1 : layers{l-1}.outputmaps  %  input map        
        k_trans = layers{l}.k{i, j}';
        der(ind+1:ind+curlen) = k_trans(:);
        ind = ind + curlen;
      end
    end  
    curlen = layers{l}.outputmaps;
    der(ind+1:ind+curlen) = layers{l}.b;
    ind = ind + curlen;
  elseif strcmp(layers{l}.type, 'f') % fully connected
    curlen = numel(layers{l}.w);
    w_trans = layers{l}.w';
    der(ind+1:ind+curlen) = w_trans(:);
    ind = ind + curlen;
    curlen = numel(layers{l}.b);
    der(ind+1:ind+curlen) = layers{l}.b;
    ind = ind + curlen;
  end;
end
%der(ind+1:ind+layers{end}.length) = layers{end}.coef;
%ind = ind + layers{end}.length;
der(ind+1:end) = [];

end