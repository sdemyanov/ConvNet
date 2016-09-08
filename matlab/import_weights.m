function layers = import_weights(layers, MCNWeights, layer_nums)

ind = 0;
for l = 1 : layer_nums
  
  if (~isfield(layers{l}, 'w'))
    continue;
  end;
  
  ind = ind + 1;
  assert(isequal(size(layers{l}.w), size(MCNWeights(ind).value)));
  layers{l}.w = MCNWeights(ind).value;
  layers{l}.w = permute(layers{l}.w, [2 1 3 4]);  
  
  if (~isfield(layers{l}, 'add_bias') || layers{l}.add_bias > 0)
    ind = ind + 1;
    assert(isequal(size(layers{l}.b), size(MCNWeights(ind).value)));
    layers{l}.b = MCNWeights(ind).value;    
  end;
  
end

end