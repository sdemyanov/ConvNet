function pred = classify_mat(layers, weights, x)

layers = cnnsetup(layers);
layers = setweights(layers, weights);

if (~iscell(x))
  copy_x = x;
  clear x;
  x{1} = copy_x;
end;

assert(length(x) == layers{1}.outputmaps, ...
  'Data must have the same number of cells as outputmaps on the first layer');  
for k = 1:length(x)
  assert(length(size(x{k})) == 3, 'The data array must have 3 dimensions');  
  assert(size(x{k}, 1) == layers{1}.mapsize(1) && ...
         size(x{k}, 2) == layers{1}.mapsize(2), ...
         'Data and the first layer must have equal sizes');
end;

[~, pred] = cnnff(layers, x, 0);

end