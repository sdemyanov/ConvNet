function num = numweights(layers)

n = numel(layers);
num = 0;
for l = 1 : n   %  layer
  
  if (isfield(layers{l}, 'w'))
    num = num + numel(layers{l}.w);    
  end;
  
  if (isfield(layers{l}, 'b'))
    num = num + numel(layers{l}.b);    
  end;
  
end

end