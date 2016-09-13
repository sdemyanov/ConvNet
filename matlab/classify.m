function pred_y = classify(layers, weights, params, test_x, test_y)

tic;
% to get NCHW layout instead of NCWH as in Matlab by default
test_x = single(permute(test_x, [2 1 3 4]));
test_y = single(permute(test_y, [2 1 3 4]));
%weights = getweights(layers);
pred_y = classify_mex(layers, weights, params, test_x, test_y);
pred_y = permute(pred_y, [2 1 3 4]);
t = toc;
disp(['Total test time: ' num2str(t)]);

end

