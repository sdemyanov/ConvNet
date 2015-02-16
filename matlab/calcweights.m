function layers = calcweights(layers, passnum)
  
n = numel(layers);
batchsize = size(layers{1}.a, 4);

for l = 1 : n
  
  if strcmp(layers{l}.type, 'n')    
    meander = layers{l}.d;
    if (layers{l}.is_dev == 1)
      meander = meander .* repmat(layers{l}.w(:, :, :, 2), [1 1 1 batchsize]);
      stdevder = layers{l-1}.a + repmat(layers{l}.w(:, :, :, 1), [1 1 1 batchsize]);
      layers{l}.dw(:, :, :, 2) = mean(layers{l}.d .* stdevder, 4);    
    end;
    layers{l}.dw(:, :, :, 1) = mean(meander, 4);
    layers{l}.dw(-layers{l}.eps < layers{l}.dw & layers{l}.dw < layers{l}.eps) = 0;

  elseif strcmp(layers{l}.type, 'c')
    a_prev = layers{l-1}.a;
    if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
      as = size(layers{l-1}.a);
      padding = layers{l}.padding;
      a_prev = zeros([as(1:2) + 2*padding size(layers{l-1}.a, 3) size(layers{l-1}.a, 4)]);
      a_prev(padding(1)+1:padding(1)+as(1), padding(2)+1:padding(2)+as(2), :, :) = layers{l-1}.a;      
    end;    
    for i = 1 : layers{l}.outputmaps        
      for j = 1 : layers{l-1}.outputmaps
        dk = filtn(a_prev(:, :, j, :), layers{l}.d(:, :, i, :), 'valid') / batchsize;
        if (passnum == 2)
          layers{l}.dk(:, :, j, i) = dk;
        elseif (passnum == 3)
          layers{l}.dk2(:, :, j, i) = dk;
        end;
      end        
    end;
    if (passnum == 2)
      layers{l}.db = squeeze(sum(sum(sum(layers{l}.d, 4), 2), 1)) / batchsize;    
      layers{l}.db = layers{l}.db * layers{l}.biascoef;
      layers{l}.db(-layers{l}.eps < layers{l}.db & layers{l}.db < layers{l}.eps) = 0;
      layers{l}.dk(-layers{l}.eps < layers{l}.dk & layers{l}.dk < layers{l}.eps) = 0;
    elseif (passnum == 3)
      layers{l}.dk2(-layers{l}.eps < layers{l}.dk2 & layers{l}.dk2 < layers{l}.eps) = 0;
    end;
    %disp('calc_weights');
    %disp(sum(layers{l}.dk(:)));
    %disp(layers{l}.dk(1, 1:5, 1, 1));

  elseif strcmp(layers{l}.type, 'f')
    dw = layers{l}.d' * layers{l}.ai / batchsize;
    if (passnum == 2)
      layers{l}.dw = dw;      
      layers{l}.db = mean(layers{l}.d, 1);
      layers{l}.db = layers{l}.db * layers{l}.biascoef;
    elseif (passnum == 3)
      layers{l}.dw2 = dw;
    end;
    if (passnum == 2) 
      layers{l}.db(-layers{l}.eps < layers{l}.db & layers{l}.db < layers{l}.eps) = 0;
      layers{l}.dw(-layers{l}.eps < layers{l}.dw & layers{l}.dw < layers{l}.eps) = 0;
    elseif (passnum == 3)
      layers{l}.dw2(-layers{l}.eps < layers{l}.dw2 & layers{l}.dw2 < layers{l}.eps) = 0;
    end;
    
    %disp('calc_weights');
    %disp(sum(layers{l}.dk(:)));
    %disp(layers{l}.dw(1, 1:5, 1, 1));
    
  end;
  
end
    
end
