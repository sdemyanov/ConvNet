function layers = backward(layers, params)

n = numel(layers);
batchsize = size(layers{1}.a, 4);

for l = n : -1 : 1
  if strcmp(layers{l}.type, 'c') || strcmp(layers{l}.type, 'f')
    if strcmp(layers{l}.function, 'soft') % for softmax
      if (~strcmp(params.lossfun, 'logreg'))
        layers{l}.d = softder(layers{l}.d, layers{l}.a);
      end;
    elseif strcmp(layers{l}.function, 'sigm') % for sigmoids
      layers{l}.d = layers{l}.d .* layers{l}.a .* (1 - layers{l}.a);          
    elseif strcmp(layers{l}.function, 'relu')
      layers{l}.d = layers{l}.d .* (layers{l}.a > 0);
    end;
    if (strcmp(layers{l}.function, 'soft') || strcmp(layers{l}.function, 'sigm'))
      layers{l}.d(-layers{l}.eps < layers{l}.d & layers{l}.d < layers{l}.eps) = 0;    
    end;
  end;
  
  if strcmp(layers{l}.type, 'c')
    d_cur = layers{l}.d;    
    if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
      ds = size(layers{l}.d); ds(end+1:4) = 1;
      padding = layers{l}.filtersize - 1 - layers{l}.padding;
      d_cur = zeros([ds(1:2) + 2*padding ds(3:4)]);
      d_cur(padding(1)+1:padding(1)+ds(1), padding(2)+1:padding(2)+ds(2), :, :) =  layers{l}.d;      
    end;
    layers{l-1}.d = zeros(size(layers{l-1}.a));
    for i = 1 : layers{l}.outputmaps        
      for j = 1 : layers{l-1}.outputmaps
        if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
          layers{l-1}.d(:, :, j, :) = layers{l-1}.d(:, :, j, :) + ...
            convn(d_cur(:, :, i, :), layers{l}.k(:, :, j, i), 'valid');
        else
          layers{l-1}.d(:, :, j, :) = layers{l-1}.d(:, :, j, :) + ...
            convn(d_cur(:, :, i, :), layers{l}.k(:, :, j, i), 'full');          
        end;
      end        
    end;
    layers{l-1}.d(-layers{l-1}.eps < layers{l-1}.d & layers{l-1}.d < layers{l-1}.eps) = 0;
    %disp(sum(layers{l-1}.d(:)));
    %disp(layers{l-1}.d(1, 1:5, 1, 1));

  elseif strcmp(layers{l}.type, 's')    
    sc = [layers{l}.scale 1 1];
    st = [layers{l}.stride 1 1];
    targsize = layers{l-1}.mapsize;
    curder = expand(layers{l}.d, sc);
    if strcmp(layers{l}.function, 'max')
      curval = expand(layers{l}.a, sc);
      prevval = stretch(layers{l-1}.a, sc, st);
      maxmat = (prevval == curval);
      %maxmat = uniq(maxmat, sc);
      curder = curder .* maxmat;
      layers{l-1}.d = shrink(curder, sc, st);
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

  elseif strcmp(layers{l}.type, 'f')
    layers{l}.di = layers{l}.d * layers{l}.w;
    if strcmp(layers{l-1}.type, 'f')      
      layers{l-1}.d = layers{l}.di;
      if (layers{l-1}.dropout > 0) % dropout
        layers{l-1}.d = layers{l-1}.d .* layers{l-1}.dropmat;
      end;
    else
      layers{l-1}.d = reshape(layers{l}.di', [layers{l-1}.mapsize layers{l-1}.outputmaps batchsize]);
    end;
    layers{l-1}.d(-layers{l-1}.eps < layers{l-1}.d & layers{l-1}.d < layers{l-1}.eps) = 0;
  end;  
end
    
end
