function layers = genweights_mat(layers, params)

rng(params.seed);

n = numel(layers);
for l = 1 : n
  if (isfield(layers{l}, 'w'))
    layers{l}.w = single((rand(size(layers{l}.w)) - 0.5) * layers{l}.initstd);    
  end;
end;

end

