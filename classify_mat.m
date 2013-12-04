function pred = classify_mat(layers, weights, x)

layers = cnnsetup(layers);
layers = setweights(layers, weights);
[~, pred] = cnnff(layers, x, 0);

end