function pred = classify_mat(layers, weights, test_x)

layers = cnnsetup(layers);
layers = setweights(layers, weights);

assert(size(test_x, 1) == layers{1}.mapsize(1) && ...
       size(test_x, 2) == layers{1}.mapsize(2), ...
       'Data and the first layer must have equal sizes');
assert(size(test_x, 3) == layers{1}.outputmaps, ...
       'The number of the input data maps must be as specified');

layers = initact(layers, test_x);
[~, pred] = forward(layers, 0);

end