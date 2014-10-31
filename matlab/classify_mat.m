function pred = classify_mat(layers, weights, params, test_x)

layers = cnnsetup(layers, 0);
layers = setweights(layers, weights);

assert(size(test_x, 1) == layers{1}.mapsize(1) && ...
       size(test_x, 2) == layers{1}.mapsize(2), ...
       'Data and the first layer must have equal sizes');
assert(size(test_x, 3) == layers{1}.outputmaps, ...
       'The number of the input data maps must be as specified');

test_x = normalize(layers{1}, test_x);     
layers = initact(layers, test_x);
[~, pred] = forward(layers, 0);

end