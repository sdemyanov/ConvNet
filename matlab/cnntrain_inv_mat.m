function [weights, trainerr] = cnntrain_inv_mat(layers, weights, train_x, train_y, params)

params = setparams(params);

assert(length(layers) == 2);
assert(length(weights) == 2);
layers{1} = cnnsetup(layers{1});
layers{1} = initact(layers{1}, train_x);
layers{2} = cnnsetup(layers{2});
layers{2} = setweights(layers{2}, weights{2});  

pixels_num = size(train_x, 4);
train_num = size(train_y, 1);
numbatches = ceil(train_num/params.batchsize);
trainerr = zeros(params.numepochs, numbatches, 2);
for i = 1 : params.numepochs
  if (params.shuffle == 0)
    kk = 1:train_num;
  else
    kk = randperm(train_num);
  end;
  for j = 1 : numbatches
    batch_ind = kk((j-1)*params.batchsize + 1 : min(j*params.batchsize, train_num));    
    batch_y = train_y(batch_ind, :);
    imlayers = cell(1, params.batchsize);
    % first pass        
    batch_w = weights{1}(:, batch_ind);        
    images = zeros([pixels_num params.batchsize]);        
    for m = 1 : params.batchsize      
      imlayers{m} = setweights(layers{1}, batch_w(:, m));
      [imlayers{m}, pred] = forward(imlayers{m}, 1);      
      images(:, m) = pred;
    end;
    images_act = reshape(images, [layers{2}{1}.mapsize layers{2}{1}.outputmaps params.batchsize]);
    layers{2} = initact(layers{2}, images_act);
    layers{2} = updateweights(layers{2}, params, i, 0); % preliminary update
    [layers{2}, pred1] = forward(layers{2}, 1);      
    %disp('pred1');
    %disp(pred1(1,1:5));
        
    % second pass
    [layers{2}, loss] = initder(layers{2}, batch_y);
    trainerr(i, j, 1) = loss;
    % disp(loss);
    %layersff{1} = imlayers;
    %layersff{2} = layers{2};    
    layers{2} = backward(layers{2});
    layers{2} = calcweights(layers{2});
    images_der = reshape(layers{2}{1}.d, [pixels_num params.batchsize]);
    for m = 1 : params.batchsize                    
      imlayers{m}{end}.d = images_der(:, m);          
      imlayers{m} = backward(imlayers{m});      
    end;        
    
    %{
    % test 0        
    layers{2} = cnnff(layers{2}, 1, 1);      
    [~, loss] = cnnder(layers{2}{end}, batch_y);
    disp(loss);
    %}
    
    % third pass    
    loss2 = 0; invnum = 0;
    invalid = zeros(params.batchsize, 1);
    for m = 1 : params.batchsize
      [imlayers{m}, curloss] = initder2(imlayers{m});
      if (curloss > 0)
        [imlayers{m}, pred] = forward(imlayers{m}, 3);      
        images(:, m) = pred;
        loss2 = loss2 + curloss;
      else
        invnum = invnum + 1;
        invalid(invnum) = m;        
      end;
    end;
    invalid(invnum+1 : end) = [];
    if (invnum < params.batchsize)
      loss2 = loss2 / (params.batchsize - invnum);
      trainerr(i, j, 2) = loss2;
      images_act = reshape(images, [layers{2}{1}.mapsize layers{2}{1}.outputmaps params.batchsize]);
      layers{2} = initact(layers{2}, images_act);
      [layers{2}, pred2] = forward(layers{2}, 3);    
    end;
    %disp(['loss2: ' num2str(loss2)]);
    layers{2} = calcweights2(layers{2}, invalid);
    %disp(['pred2: ' num2str(pred2(1,1:5))]);
    
    layers{2} = updateweights(layers{2}, params, i, 1);
    
    %{
    %test 0    
    imlayers = layersff{1};
    for m = 1 : length(layers{2})
      layers{2}{m}.a = layersff{2}{m}.a;
    end;
    layers{2}{end} = cnnder(layers{2}{end}, batch_y);
    layers{2} = cnnbp(layers{2}, 0);
    images_der = reshape(layers{2}{1}.d, [pixels_num params.batchsize]);
    loss2 = 0;
    for m = 1 : params.batchsize                    
      imlayers{m}{end}.d = images_der(:, m);          
      imlayers{m} = cnnbp(imlayers{m}, 0);
      [imlayers{m}{1}, curloss] = cnnder2(imlayers{m}{1});
      loss2 = loss2 + curloss;          
    end;        
    loss2 = loss2 / params.batchsize;
    disp(loss2); 
    
    %test 1
    alpha = params.alpha;
    params.alpha = 0;
    layers{2} = updateweights(layers{2}, params, 1);
    params.alpha = alpha;
    
    imlayers = layersff{1};
    for m = 1 : length(layers{2})
      layers{2}{m}.a = layersff{2}{m}.a;
    end;
    layers{2}{end} = cnnder(layers{2}{end}, batch_y);
    layers{2} = cnnbp(layers{2}, 0);
    images_der = reshape(layers{2}{1}.d, [pixels_num params.batchsize]);
    loss2 = 0;
    for m = 1 : params.batchsize                    
      imlayers{m}{end}.d = images_der(:, m);          
      imlayers{m} = cnnbp(imlayers{m}, 0);
      [imlayers{m}{1}, curloss] = cnnder2(imlayers{m}{1});
      loss2 = loss2 + curloss;          
    end;        
    loss2 = loss2 / params.batchsize;
    disp(loss2); 
    
    beta = params.beta;
    params.beta = 0;    
    layers{2} = updateweights(layers{2}, params, 1);
    params.beta = beta;
    
    %test 2
    imlayers = layersff{1};
    for m = 1 : length(layers{2})
      layers{2}{m}.a = layersff{2}{m}.a;
    end;
    layers{2}{end} = cnnder(layers{2}{end}, batch_y);
    layers{2} = cnnbp(layers{2}, 0);
    images_der = reshape(layers{2}{1}.d, [pixels_num params.batchsize]);
    loss2 = 0;
    for m = 1 : params.batchsize                    
      imlayers{m}{end}.d = images_der(:, m);          
      imlayers{m} = cnnbp(imlayers{m}, 0);
      [imlayers{m}{1}, curloss] = cnnder2(imlayers{m}{1});
      loss2 = loss2 + curloss;          
    end;        
    loss2 = loss2 / params.batchsize;
    disp(loss2); 
%}
    if (params.verbose == 2)
      disp(['Epoch: ' num2str(i) ', batch: ', num2str(j)]);
    end;    
  end
  if (params.verbose == 1)
    disp(['Epoch: ' num2str(i)]);
  end;
end
weights{2} = getweights(layers{2}); 
trainerr = permute(trainerr, [2 1 3]);

end
