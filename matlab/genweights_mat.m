function weights = genweights_mat(layers, params)

rng(params.seed);
layers = cnnsetup(layers, 1);
weights = getweights(layers);

end

