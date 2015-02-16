function [pred, loss, loss2]  = classify_ext_mat(layers, weights, test_x, test_y, ceps)

layers{1} = cnnsetup(layers{1}, 0);
layers{1} = initact(layers{1}, test_x);
layers{2} = cnnsetup(layers{2}, 0);
layers{2} = setweights(layers{2}, weights{2});  

assert(size(test_x, 1) == layers{1}{1}.mapsize(1) && ...
       size(test_x, 2) == layers{1}{1}.mapsize(2), ...
       'Data and the first layer must have equal sizes');
assert(size(test_x, 3) == layers{1}{1}.outputmaps, ...
       'The number of the input data maps must be as specified');

%ceps = 1e-3;
     
test_w = weights{1};
pixels_num = size(test_x, 4);
batchsize = size(test_w, 2);
imlayers = cell(1, batchsize);
images = zeros([pixels_num batchsize]);
for m = 1 : batchsize
  imlayers{m} = setweights(layers{1}, test_w(:, m));
  pix_shift = rand(1, 2) - 0.5;
  pix_shift = pix_shift / sqrt(sum(pix_shift.^2, 2));
  cur_x = test_x + ceps * repmat(pix_shift, [1 1 1 size(test_x, 4)]);  
  imlayers{m} = initact(imlayers{m}, cur_x);
  [imlayers{m}, pred] = forward(imlayers{m}, 1);      
  images(:, m) = pred;
end;
images_act = reshape(images, [layers{2}{1}.mapsize layers{2}{1}.outputmaps batchsize]);
layers{2} = initact(layers{2}, images_act);
[layers{2}, pred] = forward(layers{2}, 1);

% second pass
[layers{2}, loss] = initder(layers{2}, test_y);
layers{2} = backward(layers{2});
images_der = reshape(layers{2}{1}.d, [pixels_num batchsize]);
for m = 1 : batchsize                
  imlayers{m}{end}.d = images_der(:, m);          
  imlayers{m} = backward(imlayers{m});      
end;        

% third pass    
loss2 = 0; invnum = 0;
for m = 1 : batchsize
  [imlayers{m}, curloss] = initder2(imlayers{m});
  if (curloss > 0)
    loss2 = loss2 + curloss;
  else
    invnum = invnum + 1;
  end;
end;
if (invnum < batchsize)
  loss2 = loss2 / (batchsize - invnum);
end;
     
end