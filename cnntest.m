function [pred, err] = cnntest(layers, weights, x, y, type)

tic;
if strcmp(type, 'mexfun')
  pred = classify_mex(layers, weights, x);
elseif strcmp(type, 'matlab')
  pred = classify_mat(layers, weights, x);
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;
t = toc;
disp(['Total testing time:' num2str(t)]);

[~, ov_ind] = max(pred, [], 1);
[~, y_ind] = max(y, [], 1);
bad = find(ov_ind ~= y_ind);
err = length(bad) / size(y, 2);

end