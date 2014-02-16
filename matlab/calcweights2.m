function layers = calcweights2(layers, invalid)

n = numel(layers);
batchsize = size(layers{1}.a, 4) - length(invalid);

for l = 1 : n   %  for each layer
  
  if strcmp(layers{l}.type, 'c')
    if (batchsize == 0)
      layers{l}.dk2 = zeros(size(layers{l}.k));
      continue;
    end;
    d_cur = layers{l}.d;    
    if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
      ds = size(layers{l}.d);
      padding = layers{l}.kernelsize - 1 - layers{l}.padding;
      d_cur = zeros([ds(1:2) + 2*padding ds(3:4)]);
      d_cur(padding(1)+1:padding(1)+ds(1), padding(2)+1:padding(2)+ds(2), :, :) = layers{l}.d;      
    end;
    a_prev = layers{l-1}.a;
    if (~isempty(invalid))
      d_cur(:, :, :, invalid) = [];
      a_prev(:, :, :, invalid) = [];      
    end;    
    for i = 1 : layers{l}.outputmaps
      for j = 1 : layers{l-1}.outputmaps        
        if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
          layers{l}.dk2(:, :, j, i) = filtn(d_cur(:, :, i, :), a_prev(:, :, j, :), 'valid') / batchsize;        
        else
          layers{l}.dk2(:, :, j, i) = filtn(d_cur(:, :, i, :), a_prev(:, :, j, :), 'full') / batchsize;        
        end;
      end    
    end
    
  elseif strcmp(layers{l}.type, 'f')
    if (batchsize == 0)
      layers{l}.dw2 = zeros(size(layers{l}.w));      
      continue;
    end;
    a_cur = layers{l}.a;
    d_prev = layers{l}.di;
    if (~isempty(invalid))
      a_cur(invalid, :) = [];
      d_prev(invalid, :) = [];      
    end;    
    layers{l}.dw2 = a_cur' * d_prev / batchsize;    
  end      
end
        
end
