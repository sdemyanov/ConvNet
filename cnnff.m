function [layers, pred] = cnnff(layers, x, regime)

n = numel(layers);
batchsize = size(x, 3); % number of examples in the minibatch
layers{1}.a{1} = x;

for l = 2 : n   %  for each layer
  if strcmp(layers{l}.type, 'c')
    for i = 1 : layers{l}.outputmaps   %  for each output map
      z = layers{l}.b(i) * ones([layers{l}.mapsize batchsize]);
      for j = 1 : layers{l-1}.outputmaps   %  for each input map
        %  convolve with corresponding kernel and add to temp output map
        z = z + filtn(layers{l-1}.a{j}, layers{l}.k{i, j}, 'valid');              
      end
      if strcmp(layers{l}.function, 'sigmoid')
        layers{l}.a{i} = sigm(z);
      elseif strcmp(layers{l}.function, 'relu')
        layers{l}.a{i} = max(z, 0);
      end;      
    end    

  elseif strcmp(layers{l}.type, 's')
    s = layers{l}.scale;       
    mapsize = layers{l-1}.mapsize;
    newsize = layers{l}.mapsize;
    b = strel('rectangle', [s(1) s(2)]);
    fi = ceil((s+1)/2);
    for j = 1 : layers{l-1}.outputmaps      
      if strcmp(layers{l}.function, 'max')
        %layers{l}.a{j} = maxscale(layers{l-1}.a{j}, s);        
        a = zeros(fi(1)+s(1)*(newsize(1)-1), fi(2)+s(2)*(newsize(2)-1), batchsize);
        a(1:mapsize(1), 1:mapsize(2), :) = net.layers{l-1}.a{j};
        z = imdilate(a, b);
        net.layers{l}.a{j} = z(fi(1):s(1):end, fi(2):s(2):end, :);
      elseif strcmp(layers{l}.function, 'mean')
        a = zeros(newsize(1)*s(1), newsize(2)*s(2), batchsize);
        a(1:mapsize(1), 1:mapsize(2), :) = layers{l-1}.a{j};
        if (newsize(1) * s(1) > mapsize(1))
          a(mapsize(1)+1:end, :, :) = a(mapsize(1), :, :);
        end;
        if (newsize(2) * s(2) > mapsize(2))          
          a(:, mapsize(2)+1:end, :) = a(:, mapsize(2), :);
        end;
        z = filtn(a, ones(s) / prod(s), 'valid');        
        layers{l}.a{j} = z(1:s(1):end, 1:s(2):end, :);
      end;      
    end
    
  elseif strcmp(layers{l}.type, 'f')
    %  concatenate all end layer feature maps into vector
    if ~strcmp(layers{l-1}.type, 'f')          
      layers{l}.fv = zeros(layers{l}.weightsize(2), batchsize);
      maplen = prod(layers{l-1}.mapsize);          
      for j = 1 : layers{l-1}.outputmaps
        a_trans = permute(layers{l-1}.a{j}, [2 1 3]);
        layers{l}.fv((j-1)*maplen+1 : j*maplen, :) = reshape(a_trans, maplen, batchsize);            
      end          
    else
      layers{l}.fv = layers{l-1}.ov;
    end;

    if (regime == 0) % training
      dropmat = rand(size(layers{l}.fv));
      dropmat(dropmat < layers{l}.droprate) = 0;
      dropmat(dropmat >= layers{l}.droprate) = 1;
      layers{l}.fv = layers{l}.fv .* dropmat;
      w = layers{l}.w;
    else % testing
      w = layers{l}.w * (1 - layers{l}.droprate);
    end;    
    layers{l}.ov = w * layers{l}.fv + repmat(layers{l}.b, 1, batchsize);
    if strcmp(layers{l}.function, 'sigmoid')
      layers{l}.ov = sigm(layers{l}.ov);
    elseif strcmp(layers{l}.function, 'SVM')      
    elseif strcmp(layers{l}.function, 'relu')
      layers{l}.ov = max(layers{l}.ov, 0);
    end; 
    
  end      
end

pred = layers{n}.ov;
        
end
