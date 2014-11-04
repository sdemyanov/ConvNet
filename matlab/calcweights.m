function layers = calcweights(layers, passnum)
  
n = numel(layers);
batchsize = size(layers{1}.a, 4);

for l = 1 : n
  
  if strcmp(layers{l}.type, 'c')
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
        end;
      end        
    end;
    if (passnum == 2)
      layers{l}.db = squeeze(sum(sum(sum(layers{l}.d, 4), 2), 1)) / batchsize;    
      layers{l}.db(-layers{l}.eps < layers{l}.db & layers{l}.db < layers{l}.eps) = 0;
      layers{l}.dk(-layers{l}.eps < layers{l}.dk & layers{l}.dk < layers{l}.eps) = 0;    
    end;
    %disp('calc_weights');
    %disp(sum(layers{l}.dk(:)));
    %disp(layers{l}.dk(1, 1:5, 1, 1));

  elseif strcmp(layers{l}.type, 'f')
    dw = layers{l}.d' * layers{l}.ai / batchsize;
    if (passnum == 2)
      layers{l}.dw = dw;
      if strcmp(layers{l}.function, 'SVM')
        layers{l}.dw = layers{l}.dw + layers{l}.w / layers{l}.C;
      end;
      layers{l}.db = mean(layers{l}.d, 1);    
      layers{l}.db(-layers{l}.eps < layers{l}.db & layers{l}.db < layers{l}.eps) = 0;
      layers{l}.dw(-layers{l}.eps < layers{l}.dw & layers{l}.dw < layers{l}.eps) = 0;    
    end;
    
  end;
  
end
    
end
