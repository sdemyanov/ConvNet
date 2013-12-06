function [pred, err, bad] = cnntest(layers, weights, x, y, type)

pred = cnnclassify(layers, weights, x, type);

[~, pred_ind] = max(pred, [], 1);
[~, y_ind] = max(y, [], 1);
bad = find(pred_ind ~= y_ind);
err = length(bad) / size(y, 2);

end
