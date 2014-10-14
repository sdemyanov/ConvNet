function [layers, loss] = cnnbp(layers, y)
  
n = numel(layers);
batchsize = size(y, 1); % number of examples in the minibatch  
if (strcmp(layers{n}.function, 'SVM')) % for SVM 
  layers{n}.d = -2 * y .* max(1 - layers{n}.a .* y, 0);      
  loss = sum(sum(max(1 - layers{n}.a .* y, 0).^2)) / batchsize;
  % + 1/2 * sum(sum(layers{n}.w * layers{n}.w')) / layers{n}.C - too long
elseif (strcmp(layers{n}.function, 'sigmoid')) % for sigmoids
  layers{n}.d = layers{n}.a - y;
  loss = 1/2 * sum(layers{n}.d(:).^2) / batchsize;
end;
layers{n}.d = layers{n}.d .* repmat(layers{n}.coef, batchsize, 1); 

for l = n : -1 : 1

  if strcmp(layers{l}.type, 'c')
    if strcmp(layers{l}.function, 'sigmoid')
      layers{l}.d = layers{l}.d .* layers{l}.a .* (1 - layers{l}.a);
    elseif strcmp(layers{l}.function, 'relu')
      layers{l}.d = layers{l}.d .* (layers{l}.a > 0);
    end;    
    a_prev = layers{l-1}.a;
    d_cur = layers{l}.d;    
    if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
      as = size(layers{l-1}.a);
      padding = layers{l}.padding;
      a_prev = zeros([as(1:2) + 2*padding as(3:4)]);
      a_prev(padding(1)+1:padding(1)+as(1), padding(2)+1:padding(2)+as(2), :, :) =  layers{l-1}.a;
      ds = size(layers{l}.d);
      padding = layers{l}.kernelsize - 1 - layers{l}.padding;
      d_cur = zeros([ds(1:2) + 2*padding ds(3:4)]);
      d_cur(padding(1)+1:padding(1)+ds(1), padding(2)+1:padding(2)+ds(2), :, :) =  layers{l}.d;      
    end;    
    layers{l}.db = squeeze(sum(sum(sum(layers{l}.d, 4), 2), 1)) / batchsize;
    layers{l-1}.d = zeros(size(layers{l-1}.a));       
    for i = 1 : layers{l}.outputmaps        
      for j = 1 : layers{l-1}.outputmaps
        layers{l}.dk(:, :, j, i) = filtn(a_prev(:, :, j, :), layers{l}.d(:, :, i, :), 'valid') / batchsize;
        if (strcmp(layers{l-1}.type, 'i') || strcmp(layers{l-1}.type, 'j'))
          continue;
        end;
        if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
          layers{l-1}.d(:, :, j, :) = layers{l-1}.d(:, :, j, :) + ...
            convn(d_cur(:, :, i, :), layers{l}.k(:, :, j, i), 'valid');
        else
          layers{l-1}.d(:, :, j, :) = layers{l-1}.d(:, :, j, :) + ...
            convn(d_cur(:, :, i, :), layers{l}.k(:, :, j, i), 'full');          
        end;
      end        
    end;

  elseif strcmp(layers{l}.type, 's')
    if (strcmp(layers{l-1}.type, 'i') || strcmp(layers{l-1}.type, 'j'))
      continue;
    end;
    sc = [layers{l}.scale 1 1];
    st = [layers{l}.stride 1 1];
    targsize = layers{l-1}.mapsize;
    curder = expand(layers{l}.d, sc);
    if strcmp(layers{l}.function, 'max')
      curval = expand(layers{l}.a, sc);
      if (~isequal(sc, st))
        prevval = stretch(layers{l-1}.a, sc, st);
        maxmat = (prevval == curval);
        curder = curder .* maxmat;
        layers{l-1}.d = shrink(curder, sc, st);
      else
        maxmat = (layers{l-1}.a == curval);
        layers{l-1}.d = curder .* maxmat;
      end;
    elseif strcmp(layers{l}.function, 'mean')
      curder = curder / prod(sc);
      if (~isequal(sc, st))
        curder = shrink(curder, sc, st);
      end;
      layers{l-1}.d = curder;
    end;

    ind = (layers{l}.mapsize - 1) .* st(1:2);      
    realnum = targsize - ind;
    if (sc(1) > realnum(1))        
      extra = sum(layers{l-1}.d(targsize(1)+1:end, :, :, :), 1) / realnum(1);        
      layers{l-1}.d(targsize(1)+1:end, :, :, :) = [];
      layers{l-1}.d(ind(1)+1 : targsize(1), :, :, :) = ...
        layers{l-1}.d(ind(1)+1 : targsize(1), :, :, :) + ...
        repmat(extra, [realnum(1) 1 1 1]);
    end;
    if (sc(2) > realnum(2))
      extra = sum(layers{l-1}.d(:, targsize(2)+1:end, :, :), 2) / realnum(2);
      layers{l-1}.d(:, targsize(2)+1:end, :, :) = [];
      layers{l-1}.d(:, ind(2)+1 : targsize(2), :, :) = ...
        layers{l-1}.d(:, ind(2)+1 : targsize(2), :, :) + ...
        repmat(extra, [1 realnum(2) 1 1]);
    end;      

  elseif strcmp(layers{l}.type, 't')
    if (strcmp(layers{l-1}.type, 'i') || strcmp(layers{l-1}.type, 'j'))
      continue;
    end;
    sc = layers{l}.mapsize;
    lv = floor((sc(1) - 1)/2); hv = sc(1) - lv - 1; % lowest vertical, highest vertical
    lh = floor((sc(2) - 1)/2); hh = sc(2) - lh - 1; % lowest horizontal, highest horizontal        
    for i = 1 : layers{l-1}.outputmaps
      mi = layers{l}.mi{i};
      layers{l-1}.d{i} = zeros([layers{l-1}.mapsize batchsize]);
      for e = 1 : batchsize            
        layers{l-1}.d{i}(mi(1, e)-lv:mi(1, e)+hv, mi(2, e)-lh:mi(2, e)+hh, e) = ...
          layers{l}.d{i}(:, :, e);
      end;                    
    end

  elseif strcmp(layers{l}.type, 'f')
    if strcmp(layers{l}.function, 'SVM') % for SVM
    elseif strcmp(layers{l}.function, 'sigmoid') % for sigmoids
      layers{l}.d = layers{l}.d .* layers{l}.a .* (1 - layers{l}.a);          
    elseif strcmp(layers{l}.function, 'relu')
      layers{l}.d = layers{l}.d .* (layers{l}.a > 0);          
    end;
    layers{l}.dw = layers{l}.d' * layers{l}.ai / batchsize;
    if strcmp(layers{l}.function, 'SVM') % for SVM
      layers{l}.dw = layers{l}.dw + layers{l}.w / layers{l}.C;
    end;
    layers{l}.db = mean(layers{l}.d, 1);
    if (strcmp(layers{l-1}.type, 'i') || strcmp(layers{l-1}.type, 'j'))
      continue;
    end;
    layers{l}.di = layers{l}.d * layers{l}.w; 
    if ~strcmp(layers{l-1}.type, 'f')        
      mapsize = layers{l-1}.mapsize;
      d_trans = reshape(layers{l}.di, [batchsize mapsize(2) mapsize(1) layers{l-1}.outputmaps]);
      layers{l-1}.d = permute(d_trans, [3 2 4 1]);        
    else
      layers{l-1}.d = layers{l}.di;
    end;      
  end;
end
    
end
