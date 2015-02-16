function [weights, trainerr] = cnntrain_norm_mat(layers, weights, train_x, train_y, params)

params = setparams(params);

assert(length(layers) == 3);
assert(length(weights) == 3);
layers{1} = cnnsetup(layers{1}, 0);
layers{1} = setweights(layers{1}, weights{1});  
layers{2} = cnnsetup(layers{2}, 0);
layers{3} = cnnsetup(layers{3}, 0);
layers{3} = setweights(layers{3}, weights{3});  

mapsize = layers{1}{1}.mapsize;
pixels_num = prod(mapsize);
dimens_num = layers{1}{1}.outputmaps;
channels_num = layers{2}{end}.length;
imnetsize = [1 dimens_num 1 pixels_num];
train_num = size(train_y, 1);
numbatches = ceil(train_num/params.batchsize);
trainerr = zeros(params.numepochs, numbatches);
for i = 1 : params.numepochs
  if (params.shuffle == 0)
    kk = 1:train_num;
  else
    kk = randperm(train_num);
  end;
  for j = 1 : numbatches
    
    batch_ind = kk((j-1)*params.batchsize + 1 : min(j*params.batchsize, train_num));    
    batchsize = length(batch_ind);
    if (iscell(train_x))
      batch_x = zeros([mapsize dimens_num batchsize]);
      for m = 1 : batchsize
        cellbatch = train_x(batch_ind);
        batch_x(:, :, :, m) = cellbatch{m};
      end;
    else
      batch_x = repmat(train_x, [1 1 1 batchsize]);
    end;
    batch_y = train_y(batch_ind, :);
    batch_w = weights{2}(:, batch_ind);    
    imlayers = cell(1, batchsize);
    % first pass        
    layers{1} = initact(layers{1}, batch_x);
    layers{1} = updateweights(layers{1}, params, i, 0); % preliminary update
    [layers{1}, pred1] = forward(layers{1}, 1);
    pred1 = permute(pred1, [2 1 3 4]);
    pred1 = reshape(pred1, [pixels_num dimens_num batchsize]);
    images_act = zeros([mapsize channels_num batchsize]);
    for m = 1 : batchsize
      imlayers{m} = setweights(layers{2}, batch_w(:, m));      
      coord_act = reshape(pred1(:, :, m)', imnetsize);      
      imlayers{m} = initact(imlayers{m}, coord_act);
      [imlayers{m}, pred] = forward(imlayers{m}, 1);      
      pred = reshape(pred, [mapsize channels_num]);      
      images_act(:, :, :, m) = permute(pred, [2 1 3]);
    end;
    %load('C:\Users\sergeyd\Workspaces\Normalization\WeightsIn', 'batch_x');
    %disp(sum(abs(batch_x(:) - images_act(:))));
    layers{3} = initact(layers{3}, images_act);
    layers{3} = updateweights(layers{3}, params, i, 0); % preliminary update
    [layers{3}, pred3] = forward(layers{3}, 1);      
    %load('C:\Users\sergeyd\Workspaces\Normalization\WeightsIn', 'pred');
    %disp(sum(abs(pred3(:) - pred(:))));
    %disp(pred1(1:10,1));
        
    % second pass
    [layers{3}, loss] = initder(layers{3}, batch_y);
    trainerr(i, j) = loss;
    % disp(loss);
    layers{3} = backward(layers{3});
    layers{3} = calcweights(layers{3});
    layers{3} = updateweights(layers{3}, params, i, 1);
    %images_der = layers{3}{1}.d;
    images_der = permute(layers{3}{1}.d, [2 1 3 4]);    
    images_der = reshape(images_der, [pixels_num channels_num batchsize]);
    coord_der = zeros([mapsize dimens_num batchsize]);
    for m = 1 : batchsize                    
      imlayers{m}{end}.d = images_der(:, :, m);          
      imlayers{m} = backward(imlayers{m});
      curder = reshape(imlayers{m}{1}.d, [dimens_num pixels_num]);      
      curder = reshape(curder', [mapsize(2) mapsize(1) dimens_num]);
      %coord_der(:, :, :, m) = curder;
      coord_der(:, :, :, m) = permute(curder, [2 1 3]);      
    end;    
    layers{1}{end}.d = coord_der;
    layers{1} = backward(layers{1});
    layers{1} = calcweights(layers{1});
    layers{1} = updateweights(layers{1}, params, i, 1);
    
    if (params.verbose == 2)
      disp(['Epoch: ' num2str(i) ', batch: ', num2str(j)]);
    end;    
  end
  if (params.verbose == 1)
    disp(['Epoch: ' num2str(i)]);
  end;
end
weights{1} = getweights(layers{1}); 
weights{3} = getweights(layers{3}); 
trainerr = trainerr';

end
