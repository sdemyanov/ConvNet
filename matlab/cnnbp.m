function [layers, loss] = cnnbp(layers, y)
  n = numel(layers);
  batchsize = size(y, 2); % number of examples in the minibatch  
  if (strcmp(layers{n}.function, 'SVM')) % for SVM 
    y(y == 0) = -1; 
    layers{n}.ovd = -2 * y .* max(1 - layers{n}.ov .* y, 0);      
    loss = 1/2 * sum(sum(layers{n}.w * layers{n}.w')) / layers{n}.C + ...
       sum(sum(max(1 - layers{n}.ov .* y, 0).^2)) / batchsize;
  elseif (strcmp(layers{n}.function, 'sigmoid')) % for sigmoids
    layers{n}.ovd = layers{n}.ov - y;
    loss = 1/2 * sum(layers{n}.ovd(:).^2) / batchsize;
  end;
  layers{n}.ovd = layers{n}.ovd .* repmat(layers{end}.coef, 1, batchsize); 

  for l = n : -1 : 2

    if strcmp(layers{l}.type, 'c')        
      for j = 1 : layers{l-1}.outputmaps          
        layers{l-1}.d{j} = zeros([layers{l-1}.mapsize batchsize]);       
      end;                
      for i = 1 : layers{l}.outputmaps          
        if strcmp(layers{l}.function, 'sigmoid')
          layers{l}.d{i} = layers{l}.d{i} .* layers{l}.a{i} .* (1 - layers{l}.a{i});
        elseif strcmp(layers{l}.function, 'relu')
          layers{l}.d{i} = layers{l}.d{i} .* (layers{l}.a{i} > 0);
        end;          
        layers{l}.db(i) = sum(layers{l}.d{i}(:)) / batchsize;
        for j = 1 : layers{l-1}.outputmaps
          layers{l}.dk{i, j} = filtn(layers{l-1}.a{j}, layers{l}.d{i}, 'valid') / batchsize;
          layers{l-1}.d{j} = layers{l-1}.d{j} + filtn(layers{l}.d{i}, layers{l}.k{i, j}, 'full');          
        end        
      end;

    elseif strcmp(layers{l}.type, 's')
      s = layers{l}.scale;
      for j = 1 : layers{l-1}.outputmaps          
        targsize = layers{l-1}.mapsize;
        curder = expand(layers{l}.d{j}, [s(1) s(2) 1]);
        curder(targsize(1)+1:end, :, :) = [];
        curder(:, targsize(2)+1:end, :) = [];            
        if strcmp(layers{l}.function, 'max')
          curval = expand(layers{l}.a{j}, [s(1) s(2) 1]);
          curval(targsize(1)+1:end, :, :) = [];
          curval(:, targsize(2)+1:end, :) = [];
          maxmat = (layers{l-1}.a{j} == curval);
          layers{l-1}.d{j} = curder .* maxmat;
        elseif strcmp(layers{l}.function, 'mean')
          layers{l-1}.d{j} = curder / prod(s);
        end;          
      end

    elseif strcmp(layers{l}.type, 't')
      s = layers{l}.mapsize;
      lv = floor((s(1) - 1)/2); hv = s(1) - lv - 1; % lowest vertical, highest vertical
      lh = floor((s(2) - 1)/2); hh = s(2) - lh - 1; % lowest horizontal, highest horizontal        
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
        layers{l}.ovd = layers{l}.ovd .* layers{l}.ov .* (1 - layers{l}.ov);          
      elseif strcmp(layers{l}.function, 'relu')
        layers{l}.ovd = layers{l}.ovd .* (layers{l}.ov > 0);          
      end;
      layers{l}.dw = layers{l}.ovd * layers{l}.fv' / batchsize;
      if strcmp(layers{l}.function, 'SVM') % for SVM
        layers{l}.dw = layers{l}.dw + layers{l}.w / layers{l}.C;
      end;
      layers{l}.db = mean(layers{l}.ovd, 2);        
      layers{l}.fvd = layers{l}.w' * layers{l}.ovd; %  feature vector delta

      if ~strcmp(layers{l-1}.type, 'f')
        %  reshape feature vector deltas into output map style
        mapsize = layers{l-1}.mapsize;
        maplen = prod(mapsize);
        for j = 1 : layers{l-1}.outputmaps
          d_trans = reshape(layers{l}.fvd((j-1)*maplen+1 : j*maplen, :), ...
                            [mapsize(2) mapsize(1) batchsize]);
          layers{l-1}.d{j} = permute(d_trans, [2 1 3]);
        end
      else
        layers{l-1}.ovd = layers{l}.fvd;
      end;      
    end;
  end
    
end
