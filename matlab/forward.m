function [layers, pred] = forward(layers, passnum)

n = numel(layers);
batchsize = size(layers{1}.a, 4); % number of examples in the minibatch

for l = 1 : n   %  for each layer
  
  if strcmp(layers{l}.type, 'i')
    if (isfield(layers{l}, 'norm'))
      if (passnum == 0 || passnum == 1)
        datamean = mean(mean(layers{l}.a, 2), 1);
        layers{l}.a = layers{l}.a - repmat(datamean, [layers{l}.mapsize 1 1]);
        datanorm = sqrt(sum(sum(layers{l}.a.^2, 1), 2));
        datanorm(datanorm < layers{l}.eps) = 1;
        layers{l}.datanorm = datanorm;
      end;
      layers{l}.a = layers{l}.a ./ repmat(layers{l}.datanorm, [layers{l}.mapsize 1 1]);
      if (numel(layers{l}.norm) > 1)
        repnorm = repmat(permute(layers{l}.norm(:), [2 3 1 4]), [layers{l}.mapsize 1 batchsize]);
        layers{l}.a = layers{l}.a .* repnorm;
      else
        layers{l}.a = layers{l}.a * layers{l}.norm;
      end;
    end;      
    if (isfield(layers{l}, 'mean'))
      if (passnum == 0 || passnum == 1)      
        layers{l}.a = layers{l}.a - repmat(layers{l}.mean, [1 1 1 batchsize]);
      end;    
    end;
    if (isfield(layers{l}, 'stdev'))
      layers{l}.a = layers{l}.a ./ repmat(layers{l}.stdev, [1 1 1 batchsize]);
    end;
 
  elseif strcmp(layers{l}.type, 'c')    
    if (passnum == 0 || passnum == 1)
      layers{l}.a = repmat(permute(layers{l}.b, [3 4 1 2]), [layers{l}.mapsize 1 batchsize]);      
    elseif (passnum == 3)
      a = layers{l}.a;
      layers{l}.a = zeros([layers{l}.mapsize layers{l}.outputmaps batchsize]);      
    end;
    as = size(layers{l-1}.a); as(end+1:4) = 1;
    a_prev = layers{l-1}.a;
    if (layers{l}.padding(1) > 0 || layers{l}.padding(2) > 0)
      padding = layers{l}.padding;
      a_prev = zeros([as(1:2) + 2*padding as(3:4)]);
      a_prev(padding(1)+1:padding(1)+as(1), padding(2)+1:padding(2)+as(2), :, :) =  layers{l-1}.a;      
    end;
    for i = 1 : layers{l}.outputmaps
      for j = 1 : layers{l-1}.outputmaps        
        layers{l}.a(:, :, i, :) = layers{l}.a(:, :, i, :) + ...
          filtn(a_prev(:, :, j, :), layers{l}.k(:, :, j, i), 'valid');        
      end    
    end    

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
    
  
  elseif strcmp(layers{l}.type, 'f')    
    if (layers{l}.dropout > 0) % dropout
      if (passnum == 1) % training 1
        dropmat = rand(size(layers{l-1}.a));
        layers{l-1}.a(dropmat <= layers{l}.droprate) = 0;              
      elseif (passnum == 0) % testing      
        layers{l-1}.a = layers{l-1}.a * (1 - layers{l}.droprate);        
      end;
    end;    
    %  concatenate all end layer feature maps into vector
    if ~strcmp(layers{l-1}.type, 'f')          
      a_trans = permute(layers{l-1}.a, [4 2 1 3]);
      layers{l}.ai = reshape(a_trans, batchsize, layers{l}.weightsize(2));      
    else
      layers{l}.ai = layers{l-1}.a;
    end;
    if (passnum == 0 || passnum == 1)
      layers{l}.a = bsxfun(@plus, layers{l}.ai * layers{l}.w', layers{l}.b);      
    elseif (passnum == 3)
      a = layers{l}.a;
      layers{l}.a = layers{l}.ai * layers{l}.w';      
    end;
  end
  layers{l}.a(-layers{l}.eps < layers{l}.a & layers{l}.a < layers{l}.eps) = 0;
  
  if strcmp(layers{l}.type, 'c') || strcmp(layers{l}.type, 'f')
    if (passnum == 0 || passnum == 1)
      if strcmp(layers{l}.function, 'soft')
        layers{l}.a = soft(layers{l}.a);
      elseif strcmp(layers{l}.function, 'sigm')
        layers{l}.a = sigm(layers{l}.a);        
      elseif strcmp(layers{l}.function, 'relu')
        layers{l}.a = max(layers{l}.a, 0);
      elseif strcmp(layers{l}.function, 'SVM')
      end;
    elseif (passnum == 3)
      if strcmp(layers{l}.function, 'soft')
        layers{l}.a = softder(layers{l}.a, layers{l}.d);
      elseif strcmp(layers{l}.function, 'sigm')
        layers{l}.a = layers{l}.a .* a .* (1 - a);        
      elseif strcmp(layers{l}.function, 'relu')
        layers{l}.a = layers{l}.a .* (a > 0);
      elseif strcmp(layers{l}.function, 'SVM')
      end;
    end;
    layers{l}.a(-layers{l}.eps < layers{l}.a & layers{l}.a < layers{l}.eps) = 0;
  end;
end

pred = layers{n}.a;
        
end
