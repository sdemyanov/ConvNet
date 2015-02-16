function pred = classify_norm_mat(layers, weights, test_x)

layers{1} = cnnsetup(layers{1}, 0);
layers{1} = initact(layers{1}, test_x);
layers{2} = cnnsetup(layers{2}, 0);
layers{2} = setweights(layers{2}, weights{2});  

assert(size(test_x, 1) == layers{1}{1}.mapsize(1) && ...
       size(test_x, 2) == layers{1}{1}.mapsize(2), ...
       'Data and the first layer must have equal sizes');
assert(size(test_x, 3) == layers{1}{1}.outputmaps, ...
       'The number of the input data maps must be as specified');

test_w = weights{1};
pixels_num = size(test_x, 4);
batchsize = size(test_w, 2);
images = zeros([pixels_num batchsize]);
for m = 1 : batchsize
  layers{1} = setweights(layers{1}, test_w(:, m));
  [~, pred] = forward(layers{1}, 0);      
  images(:, m) = pred;
end;
images_act = reshape(images, [layers{2}{1}.mapsize layers{2}{1}.outputmaps batchsize]);
layers{2} = initact(layers{2}, images_act);
[layers{2}, pred] = forward(layers{2}, 0);
     
end