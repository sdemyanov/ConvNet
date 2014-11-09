function pred = cnnclassify(layers, weights, params, test_x, type)

if (length(size(test_x)) == 3)
  % insert singletone maps index
  test_x = permute(test_x, [1 2 4 3]); 
end;

tic;
if (strcmp(type, 'cpu') || strcmp(type, 'gpu'))
  pred = classify_mex(layers, weights, params, test_x);  
elseif strcmp(type, 'matlab')
  pred = classify_mat(layers, weights, params, test_x);  
else
  error('"%s" - wrong type, must be either "cpu", "gpu" or "matlab"', type);
end;
t = toc;
disp(['Total classification time: ' num2str(t)]);

end

