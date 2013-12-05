function pred = cnnclassify(layers, weights, x, type)

tic;
if strcmp(type, 'mexfun')
  pred = classify_mex(layers, weights, x);
elseif strcmp(type, 'matlab')
  pred = classify_mat(layers, weights, x);
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;
t = toc;
disp(['Total classification time: ' num2str(t)]);

end

