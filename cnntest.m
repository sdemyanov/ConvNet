function [err, bad, pred] = cnntest(layers, weights, params, test_x, test_y, type)

pred = cnnclassify(layers, weights, params, test_x, type);

[~, pred_ind] = max(pred, [], 2);
[~, y_ind] = max(test_y, [], 2);
bad = find(pred_ind ~= y_ind);
err = length(bad) / size(pred_ind, 1);

%bad = [];
%err = sum((pred - y).^2) / 2;

end
