function [layers, loss] = initder2(layers)

selfprod = sum(sum(layers{1}.d.^2));
if (selfprod < layers{1}.eps)
  layers{1}.a = zeros(size(layers{1}.a));
  loss = 0; return;
end;
pixels_num = size(layers{1}.d, 4);
products = mean(layers{1}.d, 4);
loss = sqrt(sum(products.^2, 2) / selfprod);
products = repmat(products / (pixels_num * loss), [1 1 1 pixels_num]);
layers{1}.a = (products - layers{1}.d * loss) / selfprod;
layers{1}.a(-layers{1}.eps < layers{1}.a & layers{1}.a < layers{1}.eps) = 0;

end

