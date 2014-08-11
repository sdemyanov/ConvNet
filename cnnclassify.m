function pred = cnnclassify(layers, weights_in, test_x, type)

if (length(size(test_x)) == 3)
  % insert singletone maps index
  test_x = permute(test_x, [1 2 4 3]); 
end;

tic;
if strcmp(type, 'mexfun')
  test_x = permute(test_x, [2 1 3 4]);
  for i = 1 : numel(layers)
    if (isfield(layers{i}, 'mean'))
      layers{i}.mean = permute(layers{i}.mean, [2 1 3]);
    end;
    if (isfield(layers{i}, 'stdev'))
      layers{i}.stdev = permute(layers{i}.stdev, [2 1 3]);
    end;
  end;
  pred = classify_mex(layers, weights_in, test_x);
  pred = permute(pred, [2 1]);
  %z = logsumexp(pred, 2);
  %pred = exp(bsxfun(@minus, pred, z));  
elseif strcmp(type, 'matlab')
  pred = classify_mat(layers, weights_in, test_x);
  %z = logsumexp(pred, 2);
  %pred = exp(bsxfun(@minus, pred, z));  
else
  error('"%s" - wrong type, must be either "mexfun" or "matlab"', type);
end;
t = toc;
disp(['Total classification time: ' num2str(t)]);

end

