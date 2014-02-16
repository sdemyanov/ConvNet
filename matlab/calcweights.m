function layers = calcweights(layers)
  
n = numel(layers);
batchsize = size(layers{1}.a, 4);

for l = 1 : n

  if strcmp(layers{l}.type, 'c')
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

  elseif strcmp(layers{l}.type, 'f')
    layers{l}.dw = layers{l}.d' * layers{l}.ai / batchsize;
    if strcmp(layers{l}.function, 'SVM')
      layers{l}.dw = layers{l}.dw + layers{l}.w / layers{l}.C;
    end;
    layers{l}.db = mean(layers{l}.d, 1);    
  end;
end
    
end
