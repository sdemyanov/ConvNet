function layers = cnnsetup(layers, isgen)
    
assert(strcmp(layers{1}.type, 'i'), 'The first layer must be the type of "i"');
n = numel(layers);
for l = 1 : n   %  layer
  if strcmp(layers{l}.type, 'i') % scaling
    assert(isfield(layers{l}, 'mapsize'), 'The "i" type layer must contain the "mapsize" field');
    if (~isfield(layers{l}, 'outputmaps'))
      layers{l}.outputmaps = 1;
    end;
    layers{l}.mw = single(zeros([layers{l}.mapsize layers{l}.outputmaps]));
    layers{l}.sw = single(zeros([layers{l}.mapsize layers{l}.outputmaps]));
    outputmaps = layers{l}.outputmaps;
    mapsize = layers{l}.mapsize; 
  
  elseif strcmp(layers{l}.type, 'n') % normalization    
    layers{l}.w = single(zeros([mapsize outputmaps 2]));    
    layers{l}.w(:, :, :, 2) = single(ones([mapsize outputmaps]));
    if (isfield(layers{l}, 'mean'))
      layers{l}.w(:, :, :, 1) = -layers{l}.mean;
    end;
    layers{l}.is_dev = 1;
    if (isfield(layers{l}, 'stdev'))
      if (ischar(layers{l}.stdev) && strcmp(layers{l}.stdev, 'no'))
        layers{l}.is_dev = 0;
        layers{l}.w(:, :, :, 2) = [];
      else
        layers{l}.w(:, :, :, 2) = 1 ./ layers{l}.stdev;
      end;
    end;    
    layers{l}.dw = single(zeros(size(layers{l}.w)));
    layers{l}.dw2 = single(zeros(size(layers{l}.w)));
    layers{l}.dwp = single(zeros(size(layers{l}.w)));
    layers{l}.gw = single(ones(size(layers{l}.w)));
  
  elseif strcmp(layers{l}.type, 'j') % scaling
    assert(isfield(layers{l}, 'mapsize'), 'The "j" type layer must contain the "mapsize" field');    
    mapsize = layers{l}.mapsize;    
  
  elseif strcmp(layers{l}.type, 's') % scaling
    assert(isfield(layers{l}, 'scale'), 'The "s" type layer must contain the "scale" field');
    if (~isfield(layers{l}, 'function'))
      layers{l}.function = 'mean';
    end;
    if ~strcmp(layers{l}.function, 'max') && ~strcmp(layers{l}.function, 'mean')
      error('"%s" - unknown function for the layer %d', layers{l}.function, l);
    end;
    if (~isfield(layers{l}, 'stride'))
      layers{l}.stride = layers{l}.scale;
    end;
    mapsize = ceil(mapsize ./ layers{l}.stride);
    
  elseif strcmp(layers{l}.type, 'c') % convolutional
    assert(isfield(layers{l}, 'filtersize'), 'The "c" type layer must contain the "filtersize" field');
    assert(isfield(layers{l}, 'outputmaps'), 'The "c" type layer must contain the "outputmaps" field');
    if (~isfield(layers{l}, 'function'))
      layers{l}.function = 'relu';
    end;
    if (~strcmp(layers{l}.function, 'sigm') && ...
       ~strcmp(layers{l}.function, 'relu')) % REctified Linear Unit
      error('"%s" - unknown function for the layer %d', layers{l}.function, l);
    end;
    if (~isfield(layers{l}, 'padding'))
      layers{l}.padding = [0 0];
    end;
    if (~isfield(layers{l}, 'initstd'))
      layers{l}.initstd = 0.01;
    end;
    if (~isfield(layers{l}, 'biascoef'))
      layers{l}.biascoef = 1;
    end;
    
    %fan_in = outputmaps * layers{l}.filtersize(1) *  layers{l}.filtersize(2);
    %fan_out = layers{l}.outputmaps * layers{l}.filtersize(1) * layers{l}.filtersize(2);
    %rand_coef = 2 * sqrt(6 / (fan_in + fan_out));
    layers{l}.k = single(zeros([layers{l}.filtersize outputmaps layers{l}.outputmaps]));
    layers{l}.dk = single(zeros([layers{l}.filtersize outputmaps layers{l}.outputmaps]));
    layers{l}.dk2 = single(zeros([layers{l}.filtersize outputmaps layers{l}.outputmaps]));
    layers{l}.dkp = single(zeros([layers{l}.filtersize outputmaps layers{l}.outputmaps]));
    layers{l}.gk = single(ones([layers{l}.filtersize outputmaps layers{l}.outputmaps]));
    if (isgen)
      layers{l}.k = single(randn([layers{l}.filtersize outputmaps layers{l}.outputmaps]) * layers{l}.initstd);
    else
      layers{l}.k = single(zeros([layers{l}.filtersize outputmaps, layers{l}.outputmaps]));
    end;
    layers{l}.b = single(zeros(layers{l}.outputmaps, 1));
    layers{l}.db = single(zeros(layers{l}.outputmaps, 1));    
    layers{l}.db2 = single(zeros(layers{l}.outputmaps, 1));    
    layers{l}.dbp = single(zeros(layers{l}.outputmaps, 1));
    layers{l}.gb = single(ones(layers{l}.outputmaps, 1));
    mapsize = mapsize + 2*layers{l}.padding - layers{l}.filtersize + 1;
    outputmaps = layers{l}.outputmaps;

  elseif strcmp(layers{l}.type, 'f') % fully connected
    if (~isfield(layers{l}, 'dropout'))
      layers{l}.dropout = 0; % no dropout
    end;
    if (~isfield(layers{l}, 'function'))
      layers{l}.function = 'relu';
    end;
    if (~strcmp(layers{l}.function, 'relu') && ...
        ~strcmp(layers{l}.function, 'sigm') && ...
        ~strcmp(layers{l}.function, 'soft'))
      error('"%s" - unknown function for the layer %d', layers{l}.function, l);
    end;
    if (~isfield(layers{l}, 'initstd'))
      layers{l}.initstd = 0.1;
    end;
    if (~isfield(layers{l}, 'biascoef'))
      layers{l}.biascoef = 1;
    end;
    assert(isfield(layers{l}, 'length'), 'The "f" type layer must contain the "length" field');      
    weightsize(1) = layers{l}.length;
    if ~strcmp(layers{l-1}.type, 'f')
      maplen = prod(layers{l-1}.mapsize);        
      weightsize(2) = maplen * outputmaps;
    else
      weightsize(2) = layers{l-1}.length;
    end;
    layers{l}.weightsize = weightsize; 
    if (isgen)
      layers{l}.w = single(randn(weightsize) * layers{l}.initstd);      
    else
      layers{l}.w = single(zeros(weightsize));
    end;
    layers{l}.dw = single(zeros(weightsize));
    layers{l}.dw2 = single(zeros(weightsize));
    layers{l}.dwp = single(zeros(weightsize));
    layers{l}.gw = single(ones(weightsize));

    layers{l}.b = single(zeros(1, weightsize(1)));
    layers{l}.db = single(zeros(1, weightsize(1)));    
    layers{l}.db2 = single(zeros(1, weightsize(1)));    
    layers{l}.dbp = single(zeros(1, weightsize(1)));
    layers{l}.gb = single(ones(1, weightsize(1)));      
    mapsize = [0 0];
    outputmaps = 0;      
  else
    error('"%s" - unknown type of the layer %d', layers{l}.type, l);
  end
  if (~isfield(layers{l}, 'function'))
    layers{l}.function = 'none';
  end;
  layers{l}.outputmaps = outputmaps;
  layers{l}.mapsize = mapsize;   
  layers{l}.eps = 1e-6;
end
%assert(strcmp(layers{n}.type, 'f'), 'The last layer must be the type of "f"'); 
  
end
