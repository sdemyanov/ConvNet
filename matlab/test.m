function [err, bad, pred_y] = test(layers, weights, params, test_x, test_y)

pred_y = classify(layers, weights, params, test_x, test_y);

[~, pred_ind] = max(pred_y, [], 3);
[~, test_ind] = max(test_y, [], 3);
bad = (pred_ind ~= test_ind);
err = mean(bad(:));

end
