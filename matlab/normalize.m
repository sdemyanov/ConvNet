function [train_x] = normalize(firstlayer, train_x)

if (isfield(firstlayer, 'norm'))
  datanorm = sqrt(sum(sum(sum(train_x.^2, 1), 2), 3));
  datanorm(datanorm <= firstlayer.eps) = firstlayer.norm;
  train_x = train_x ./ repmat(datanorm, [firstlayer.mapsize firstlayer.outputmaps 1]);
  train_x = train_x * firstlayer.norm;
end;

end

