function [firstlayer] = initnorm(firstlayer, train_x)

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

