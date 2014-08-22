function [firstlayer] = initnorm(firstlayer, train_x)

if (~isfield(firstlayer, 'mean') && ~isfield(firstlayer, 'maxdev')) 
  return;
end;
if (isfield(firstlayer, 'norm'))
  datanorm = sqrt(sum(sum(sum(train_x.^2, 1), 2), 3));
  datanorm(datanorm <= firstlayer.eps) = firstlayer.norm;
  train_x = train_x ./ repmat(datanorm, [firstlayer.mapsize firstlayer.outputmaps 1]);
  train_x = train_x * firstlayer.norm;      
end;      
if (isfield(firstlayer, 'mean'))
  firstlayer.mw = firstlayer.mean - mean(train_x, 4);
end;
if (isfield(firstlayer, 'maxdev'))
  stdev = std(train_x, 1, 4);
  stdev(stdev <= firstlayer.maxdev) = firstlayer.maxdev;
  firstlayer.sw = firstlayer.maxdev * ones([firstlayer.mapsize firstlayer.outputmaps]);
  firstlayer.sw = firstlayer.sw ./ stdev;
end;

end

