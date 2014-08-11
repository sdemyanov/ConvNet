function layers = calcweights(layers)
  
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
      a_prev = zeros([as(1:2) + 2*padding as(3:4)]);
      a_prev(padding(1)+1:padding(1)+as(1), padding(2)+1:padding(2)+as(2), :, :) =  layers{l-1}.a;      
    end;    
    layers{l}.db = squeeze(sum(sum(sum(layers{l}.d, 4), 2), 1)) / batchsize;
    for i = 1 : layers{l}.outputmaps        
      for j = 1 : layers{l-1}.outputmaps
        layers{l}.dk(:, :, j, i) = filtn(a_prev(:, :, j, :), layers{l}.d(:, :, i, :), 'valid') / batchsize;                
      end        
    end;
    layers{l}.db(-layers{l}.eps < layers{l}.db & layers{l}.db < layers{l}.eps) = 0;
    layers{l}.dk(-layers{l}.eps < layers{l}.dk & layers{l}.dk < layers{l}.eps) = 0;

  elseif strcmp(layers{l}.type, 'f')
    if (layers{l}.dropout > 0)
      dividers_bias = sum(layers{l}.dropmat_bias, 1);
      dividers_bias(dividers_bias == 0) = 1;
      layers{l}.db = sum(layers{l}.d, 1) ./ dividers_bias;
      
      dividers = sum(layers{l}.dropmat, 3);
      dividers(dividers == 0) = 1;
      layers{l}.dw = maskprod(layers{l}.d, 1, layers{l}.ai, 0, layers{l}.dropmat);
      layers{l}.dw = layers{l}.dw ./ dividers;      
    else
      layers{l}.db = mean(layers{l}.d, 1);
      layers{l}.dw = layers{l}.d' * layers{l}.ai / batchsize;
    end;
    if strcmp(layers{l}.function, 'SVM')
      layers{l}.dw = layers{l}.dw + layers{l}.w / layers{l}.C;
    end;    
    layers{l}.db(-layers{l}.eps < layers{l}.db & layers{l}.db < layers{l}.eps) = 0;
    layers{l}.dw(-layers{l}.eps < layers{l}.dw & layers{l}.dw < layers{l}.eps) = 0;
  end;
  
end
    
end
