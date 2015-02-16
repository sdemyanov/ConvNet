function [layers, loss] = initder_dir_coord_inv(layers)

activ = zeros(size(layers{1}.a));
pixels_num = size(layers{1}.d, 4);
selfprod = sum(sum(layers{1}.d.^2));
if (selfprod < layers{1}.eps)
  layers{1}.a = zeros(size(layers{1}.a));
  loss = 0; return;
end;
loss = 0;

norm_shift = sqrt(pixels_num);
products_shift = sum(layers{1}.d, 4) ./ norm_shift;
activ = activ + repmat(products_shift ./ norm_shift, [1 1 1 pixels_num]);
loss = loss + sum(products_shift.^2);

vectors_scale = layers{1}.a - repmat(mean(layers{1}.a, 4), [1 1 1 pixels_num]);
norm_scale_sq = sum(vectors_scale.^2, 4);
norm_scale = sqrt(norm_scale_sq);
products_scale = sum(layers{1}.d .* vectors_scale, 4) ./ norm_scale;
activ = activ + vectors_scale .* repmat(products_scale ./ norm_scale, [1 1 1 pixels_num]);
loss = loss + sum(products_scale.^2);

vectors_rot = layers{1}.a(:, [2 1], :, :);
vectors_rot(:, 2, :, :) = -vectors_rot(:, 2, :, :);
vectors_rot = vectors_rot - repmat(mean(vectors_rot, 4), [1 1 1 pixels_num]);
rot_scale_prod = sum(vectors_rot .* vectors_scale, 4) ./ norm_scale_sq;
vectors_rot = vectors_rot - vectors_scale .* repmat(rot_scale_prod, [1 1 1 pixels_num]);
norm_rot = sqrt(sum(sum(vectors_rot.^2)));
products_rot = sum(sum(layers{1}.d .* vectors_rot)) / norm_rot;
activ = activ + vectors_rot * products_rot / norm_rot;
loss = loss + products_rot^2;

loss = sqrt(loss / selfprod);
layers{1}.a = (activ / loss - layers{1}.d * loss) / selfprod;
layers{1}.a(-layers{1}.eps < layers{1}.a & layers{1}.a < layers{1}.eps) = 0;

%ind = 1 : size(layers{1}.a, 1) * size(layers{1}.a, 2);
%ind = reshape(ind, size(layers{1}.a, 1), size(layers{1}.a, 2))';
%a = layers{1}.a(:,:,:,ind(:));

end

