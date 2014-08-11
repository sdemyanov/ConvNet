function weights = genweights_mat(layers, seed)

rng(seed);
layers = cnnsetup(layers, 1);
weights = getweights(layers);

end

