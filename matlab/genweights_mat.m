function weights = genweights_mat(layers)

layers = cnnsetup(layers, 1);
weights = getweights(layers);

end

