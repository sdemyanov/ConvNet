function [layers, pred] = forward(layers, passnum)

n = numel(layers);
batchsize = size(layers{1}.a, 4); % number of examples in the minibatch

for l = 1 : n   %  for each layer
  
  if strcmp(layers{l}.type, 'i')
    if (passnum == 3)
      continue;
    end;   
    if (isfield(layers{l}, 'mean'))
      layers{l}.a = layers{l}.a + repmat(layers{l}.mw, [1 1 1 batchsize]);      
    end;
    if (isfield(layers{l}, 'maxdev'))
      layers{l}.a = layers{l}.a .* repmat(layers{l}.sw, [1 1 1 batchsize]);
    end;
    
  elseif strcmp(layers{l}.type, 'n')
    layers{l}.a = layers{l-1}.a + repmat(layers{l}.w(:, :, :, 1), [1 1 1 batchsize]);
    if (layers{l}.is_dev == 1)
      layers{l}.a = layers{l}.a .* repmat(layers{l}.w(:, :, :, 2), [1 1 1 batchsize]);
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
    %{
    disp(layers{l-1}.a(1, 1:5, 1, 1));
    disp(layers{l}.k(1, 1:4, 1, 1));
    disp(layers{l}.k(1:4, 1, 1, 1));
    disp(layers{l}.a(1, 1:5, 1, 1));    
    %}

  elseif strcmp(layers{l}.type, 's')
    sc = [layers{l}.scale 1 1];
    st = [layers{l}.stride 1 1];
    mapsize = layers{l-1}.mapsize;
    newsize = layers{l}.mapsize;
    b = strel('rectangle', [sc(1) sc(2)]);
    fi = ceil((sc+1)/2);    
    if strcmp(layers{l}.function, 'max')
      if (passnum == 0 || passnum == 1)
        %layers{l}.a{j} = maxscale(layers{l-1}.a{j}, s);
        a = double(zeros(fi(1)+st(1)*(newsize(1)-1), fi(2)+st(2)*(newsize(2)-1), layers{l}.outputmaps, batchsize));
        a(1:mapsize(1), 1:mapsize(2), :, :) = layers{l-1}.a;
        z = imdilate(a, b);
        layers{l}.a = z(fi(1):st(1):end, fi(2):st(2):end, :, :);
      elseif (passnum == 3)        
        curval = expand(layers{l}.d, sc);
        prevval = stretch(layers{l-1}.d, sc, st);
        maxmat = (prevval == curval);
        %maxmat = uniq(maxmat, sc);
        curder = stretch(layers{l-1}.a, sc, st);        
        curder = curder .* maxmat;
        z = convn(curder, ones(sc(1:2)), 'valid');       
        layers{l}.a = z(1:sc(1):end, 1:sc(2):end, :, :);        
      end;      
      
    elseif strcmp(layers{l}.function, 'mean')
      a = double(zeros([(newsize-1).*st(1:2)+sc(1:2) layers{l}.outputmaps batchsize]));
      a(1:mapsize(1), 1:mapsize(2), :, :) = layers{l-1}.a;
      if (size(a, 1) > mapsize(1))
        meanvals = mean(a((newsize(1)-1)*st(1)+1 : mapsize(1), :, :, :), 1);
        extind = mapsize(1)+1 : size(a, 1);
        a(extind, :, :, :) = repmat(meanvals, [length(extind) 1 1 1]);
      end;
      if (size(a, 2) > mapsize(2))         
        meanvals = mean(a(:, (newsize(2)-1)*st(2)+1 : mapsize(2), :, :), 2);
        extind = mapsize(2)+1 : size(a, 2);
        a(:, extind, :, :) = repmat(meanvals, [1 length(extind) 1 1]);
      end;
      z = convn(a, ones(sc(1:2)) / prod(sc), 'valid');        
      layers{l}.a = z(1:st(1):end, 1:st(2):end, :, :);
    end;
    
  
  elseif strcmp(layers{l}.type, 'f')    
    %  concatenate all end layer feature maps into vector
    if ~strcmp(layers{l-1}.type, 'f')          
      %a_trans = permute(layers{l-1}.a, [4 1 2 3]);
      %disp('prev_layer->activ_mat');
      %disp(a_trans(1, 1:5, 1, 1));
      %disp('weights');
      %disp(layers{l}.w(1, 1:5));
      layers{l}.ai = reshape(layers{l-1}.a, layers{l}.weightsize(2), batchsize)';      
    else
      layers{l}.ai = layers{l-1}.a;
    end;
    if (layers{l}.dropout > 0) % dropout
      if (passnum == 1) % training 1
        dropmat = rand([size(layers{l}.w) batchsize]);
        dropmat(dropmat < layers{l}.dropout) = 0;
        dropmat(dropmat > 0) = 1;
        layers{l}.dropmat = dropmat;        
        layers{l}.a = maskprod(layers{l}.ai, 0, layers{l}.w, 1, dropmat);        
        
        dropmat_bias = rand([batchsize size(layers{l}.b, 2)]);
        dropmat_bias(dropmat_bias < layers{l}.dropout) = 0;
        dropmat_bias(dropmat_bias > 0) = 1;
        layers{l}.dropmat_bias = dropmat_bias;        
        biases = repmat(layers{l}.b, [batchsize 1]) .* dropmat_bias;
        layers{l}.a = layers{l}.a + biases;
      elseif (passnum == 3)
        a = layers{l}.a;
        layers{l}.a = maskprod(layers{l}.ai, 0, layers{l}.w, 1, layers{l}.dropmat);        
      elseif (passnum == 0) % testing      
        layers{l-1}.a = layers{l-1}.a * (1 - layers{l}.dropout);
        biases = layers{l}.b * (1 - layers{l}.dropout);
        layers{l}.a = bsxfun(@plus, layers{l}.ai * layers{l}.w', biases);      
      end;    
    else
      if (passnum == 0 || passnum == 1)
        layers{l}.a = bsxfun(@plus, layers{l}.ai * layers{l}.w', layers{l}.b);      
      elseif (passnum == 3)
        a = layers{l}.a;
        layers{l}.a = layers{l}.ai * layers{l}.w';      
      end;
    end;
    %disp('activ_mat');
    %disp(layers{l}.a(1, 1:5, 1, 1));
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
