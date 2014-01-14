function [layers, pred] = cnnff(layers, x, regime)

n = numel(layers);
batchsize = size(x, 4); % number of examples in the minibatch
layers{1}.a = x;

for l = 1 : n   %  for each layer
  
  if strcmp(layers{l}.type, 'i')
    if (layers{l}.norm > 0 )
      datamean = mean(mean(layers{l}.a, 2), 1);
      layers{l}.a = layers{l}.a - repmat(datamean, [layers{l}.mapsize 1 1]);
      datanorm = sqrt(sum(sum(layers{l}.a.^2, 1), 2));
      datanorm(datanorm < 1e-8) = 1;
      layers{l}.a = layers{l}.norm * layers{l}.a ./ repmat(datanorm, [layers{l}.mapsize 1 1]);
    end;
    if (sum(sum(layers{l}.mean)) > 0)
      layers{l}.a = layers{l}.a - repmat(layers{l}.mean, [1 1 1 batchsize]);
    end;
    if (sum(sum(layers{l}.stdev)) > 0)
      layers{l}.a = layers{l}.a ./ repmat(layers{l}.stdev, [1 1 1 batchsize]);
    end;
  
  elseif strcmp(layers{l}.type, 'j')
    assert(sum(mod(layers{l-1}.mapsize - layers{l}.mapsize, 2)) == 0);
    layers{l}.a = layers{l-1}.a;
    delsize = floor((layers{l-1}.mapsize - layers{l}.mapsize) ./ 2);
    delind = [1:delsize(1) size(layers{l-1}.a, 1)-delsize(1)+1:size(layers{l-1}.a, 1)];
    layers{l}.a(delind, :, :, :) = [];
    delind = [1:delsize(2) size(layers{l-1}.a, 2)-delsize(2)+1:size(layers{l-1}.a, 2)];
    layers{l}.a(:, delind, :, :) = [];    
  
  elseif strcmp(layers{l}.type, 'c')
    as = size(layers{l-1}.a);
    a_prev = layers{l-1}.a;
    if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
      padding = layers{l}.padding;
      a_prev = zeros([as(1:2) + 2*padding as(3:4)]);
      a_prev(padding(1)+1:padding(1)+as(1), padding(2)+1:padding(2)+as(2), :, :) =  layers{l-1}.a;      
    end;
    layers{l}.a = repmat(permute(layers{l}.b, [2 3 1 4]), [layers{l}.mapsize 1 batchsize]);
    for i = 1 : layers{l}.outputmaps   %  for each output map      
      for j = 1 : layers{l-1}.outputmaps   %  for each input map
        %  convolve with corresponding kernel and add to temp output map
        layers{l}.a(:, :, i, :) = layers{l}.a(:, :, i, :) + ...
          filtn(a_prev(:, :, j, :), layers{l}.k(:, :, j, i), 'valid');              
      end    
    end      
    if strcmp(layers{l}.function, 'sigmoid')
      layers{l}.a = sigm(layers{l}.a);
    elseif strcmp(layers{l}.function, 'relu')
      layers{l}.a = max(layers{l}.a, 0);
    end;      

  elseif strcmp(layers{l}.type, 's')
    sc = layers{l}.scale;
    st = layers{l}.stride;
    mapsize = layers{l-1}.mapsize;
    newsize = layers{l}.mapsize;
    b = strel('rectangle', [sc(1) sc(2)]);
    fi = ceil((sc+1)/2);    
    if strcmp(layers{l}.function, 'max')
      %layers{l}.a{j} = maxscale(layers{l-1}.a{j}, s);        
      a = double(zeros(fi(1)+st(1)*(newsize(1)-1), fi(2)+st(2)*(newsize(2)-1), layers{l}.outputmaps, batchsize));
      a(1:mapsize(1), 1:mapsize(2), :, :) = layers{l-1}.a;
      z = imdilate(a, b);
      layers{l}.a = z(fi(1):st(1):end, fi(2):st(2):end, :, :);
    elseif strcmp(layers{l}.function, 'mean')
      a = double(zeros([(newsize-1).*st+sc layers{l}.outputmaps batchsize]));
      a(1:mapsize(1), 1:mapsize(2), :, :) = layers{l-1}.a;
      if (size(a, 1) > mapsize(1))
        a(mapsize(1)+1:end, :, :, :) = mean(a((newsize(1)-1)*st(1)+1 : mapsize(1), :, :, :), 1);
      end;
      if (size(a, 2) > mapsize(2))          
        a(:, mapsize(2)+1:end, :, :) = mean(a(:, (newsize(2)-1)*st(2)+1 : mapsize(2), :, :), 2);
      end;
      z = filtn(a, ones(sc) / prod(sc), 'valid');        
      layers{l}.a = z(1:st(1):end, 1:st(2):end, :, :);
    end;
    
  elseif strcmp(layers{l}.type, 't') % neighborhood of maximum        
    sc = layers{l}.mapsize; % size vertical
    lv = floor((sc(1) - 1)/2); hv = sc(1) - lv - 1; % lowest vertical, highest vertical
    lh = floor((sc(2) - 1)/2); hh = sc(2) - lh - 1; % lowest horizontal, highest horizontal
    for j = 1 : layers{l-1}.outputmaps
      layers{l}.a{j} = zeros(sc(1), sc(2), batchsize);
      layers{l}.mi{j} = zeros(2, batchsize); % maximum indices
      for e = 1 : batchsize
        curval = layers{l-1}.a{j}(:, :, e);
        sc = size(curval); % size current
        while (sc(1) < sc(1))
          curval(end+1, :) = 0; %curval(end, :);
          sc(1) = sc(1) + 1;
        end;
        while (sc(2) < sc(2))
          curval(:, end+1) = 0; %curval(:, end);
          sc(2) = sc(2) + 1;
        end;
        availmax = curval(1+lv:sc(1)-hv, 1+lh:sc(2)-hh);
        % maximum vertical, maximum horizontal
        [mv, mh] = find(availmax == max(availmax(:)), 1, 'first');
        mv = mv + lv; mh = mh + lh;
        layers{l}.a{j}(:, :, e) = curval(mv-lv:mv+hv, mh-lh:mh+hh);
        layers{l}.mi{j}(:, e) = [mv; mh];                                  
      end;        
    end

  elseif strcmp(layers{l}.type, 'f')
    
    if (layers{l}.droprate > 0) % dropout
      if (regime == 1) % training      
        dropmat = rand(size(layers{l-1}.a));
        layers{l-1}.a(dropmat <= layers{l}.droprate) = 0;        
      else % testing      
        layers{l-1}.a = layers{l-1}.a * (1 - layers{l}.droprate);        
      end;
    end;
    
    %  concatenate all end layer feature maps into vector
    if ~strcmp(layers{l-1}.type, 'f')          
      a_trans = permute(layers{l-1}.a, [4 2 1 3]);
      layers{l}.ai = reshape(a_trans, batchsize, layers{l}.weightsize(1));      
    else
      layers{l}.ai = layers{l-1}.a;
    end;
   
    layers{l}.a = bsxfun(@plus, layers{l}.ai * layers{l}.w, layers{l}.b);
    if strcmp(layers{l}.function, 'sigmoid')
      layers{l}.a = sigm(layers{l}.a);
    elseif strcmp(layers{l}.function, 'relu')
      layers{l}.a = max(layers{l}.a, 0);
    elseif strcmp(layers{l}.function, 'SVM')    
    end; 
    
  end      
end

pred = layers{n}.a;
        
end
