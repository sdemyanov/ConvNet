function [layers, loss] = initder_dir_int_inv(layers)

selfprod = squeeze(sum(sum(sum(layers{1}.d.^2, 1), 2), 3));
sig_ind = find(selfprod >= layers{1}.eps);
sig_num = length(sig_ind);
selfprod = selfprod(sig_ind);
sigder = layers{1}.d(:, :, :, sig_ind);

activ_size = [size(layers{1}.a, 1) size(layers{1}.a, 2) size(layers{1}.a, 3)];
batchsize = size(layers{1}.a, 4);
activ = zeros([activ_size sig_num]);
sigloss = zeros(sig_num, 1);

vectors_bright = ones([activ_size sig_num]);
norm_int = sqrt(sum(sum(sum(vectors_bright.^2, 1), 2), 3));
vectors_bright = vectors_bright ./ repmat(norm_int, [activ_size 1]);
products_bright = sum(sum(sum(sigder .* vectors_bright, 1), 2), 3);
activ = activ + vectors_bright .* repmat(products_bright, [activ_size 1]);
sigloss = sigloss + squeeze(products_bright.^2);

vectors_bright = layers{1}.a(:, :, :, sig_ind);
% to make it orthogonal to uniform vector
vectors_mean = sum(sum(sum(vectors_bright, 1), 2), 3) / prod(activ_size);
vectors_bright = vectors_bright - repmat(vectors_mean, [activ_size 1]);
norm_int = sqrt(sum(sum(sum(vectors_bright.^2, 1), 2), 3));
vectors_bright = vectors_bright ./ repmat(norm_int, [activ_size 1]);
products_bright = sum(sum(sum(sigder .* vectors_bright, 1), 2), 3);
activ = activ + vectors_bright .* repmat(products_bright, [activ_size 1]);
sigloss = sigloss + squeeze(products_bright.^2);

sigloss = sqrt(sigloss ./ selfprod);
loss = zeros(batchsize, 1);
loss(sig_ind) = sigloss;
sigloss = repmat(shiftdim(sigloss, -3), [activ_size 1]);
selfprod = repmat(shiftdim(selfprod, -3), [activ_size 1]);
activ = (activ ./ sigloss - sigder .* sigloss) ./ selfprod;
layers{1}.a = zeros(size(layers{1}.a));
layers{1}.a(:, :, :, sig_ind) = activ;

layers{1}.a(-layers{1}.eps < layers{1}.a & layers{1}.a < layers{1}.eps) = 0;

%ind = 1 : size(layers{1}.a, 1) * size(layers{1}.a, 2);
%ind = reshape(ind, size(layers{1}.a, 1), size(layers{1}.a, 2))';
%a = layers{1}.a(:,:,:,ind(:));

end
