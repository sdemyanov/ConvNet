function layers = import_weights(layers, MCNWeights, layer_nums)

layers = setup(layers);
ind = 0;
for l = 1 : layer_nums
  
  if (isfield(layers{l}, 'weightsize'))
    ind = ind + 1;
    weights = permute(MCNWeights(ind).value, [2 1 3 4]);
    assert(isequal(layers{l}.weightsize, size(weights)));
    layers{l}.w = weights;
  end;
  
  if (isfield(layers{l}, 'biassize'))
    ind = ind + 1;
    biases = MCNWeights(ind).value;
    assert(isequal(layers{l}.biassize, size(biases)));
    layers{l}.b = biases;
  end;
  
end

end
