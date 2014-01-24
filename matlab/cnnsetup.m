function layers = cnnsetup(layers)
    
assert(strcmp(layers{1}.type, 'i'), 'The first layer must be the type of "i"');
n = numel(layers);
for l = 1 : n   %  layer
  if strcmp(layers{l}.type, 'i') % scaling
    assert(isfield(layers{l}, 'mapsize'), 'The "i" type layer must contain the "mapsize" field');
    if (~isfield(layers{l}, 'outputmaps'))
      layers{l}.outputmaps = 1;
    end;
    if (~isfield(layers{l}, 'norm'))
      layers{l}.norm = 0;
    end;
    if (~isfield(layers{l}, 'mean'))
      layers{l}.mean = zeros([layers{l}.mapsize layers{l}.outputmaps]);
    end;
    if (~isfield(layers{l}, 'stdev'))
      layers{l}.stdev = zeros([layers{l}.mapsize layers{l}.outputmaps]);
    end;
    outputmaps = layers{l}.outputmaps;
    mapsize = layers{l}.mapsize; 
  
  elseif strcmp(layers{l}.type, 'j') % scaling
    assert(isfield(layers{l}, 'mapsize'), 'The "j" type layer must contain the "mapsize" field');    
    mapsize = layers{l}.mapsize;    
  
  elseif strcmp(layers{l}.type, 's') % scaling
    assert(isfield(layers{l}, 'scale'), 'The "s" type layer must contain the "scale" field');
    if (~isfield(layers{l}, 'function'))
      layers{l}.function = 'max';
    end;
    if ~strcmp(layers{l}.function, 'max') && ~strcmp(layers{l}.function, 'mean')
      error('"%s" - unknown function for the layer %d', layers{l}.function, l);
    end;
    if (~isfield(layers{l}, 'stride'))
      layers{l}.stride = layers{l}.scale;
    end;
    mapsize = ceil(mapsize ./ layers{l}.stride);
    
  elseif strcmp(layers{l}.type, 't') % neighborhood of maximum      
    assert(isfield(layers{l}, 'mapsize'), 'The "t" type layer must contain the "mapsize" field');
    mapsize = layers{l}.mapsize; 

  elseif strcmp(layers{l}.type, 'c') % convolutional
    assert(isfield(layers{l}, 'kernelsize'), 'The "c" type layer must contain the "kernelsize" field');
    assert(isfield(layers{l}, 'outputmaps'), 'The "c" type layer must contain the "outputmaps" field');
    if (~isfield(layers{l}, 'function'))
      layers{l}.function = 'sigmoid';
    end;
    if ~strcmp(layers{l}.function, 'sigmoid') && ...
       ~strcmp(layers{l}.function, 'relu') % REctified Linear Unit
      error('"%s" - unknown function for the layer %d', layers{l}.function, l);
    end;
    if (~isfield(layers{l}, 'padding'))
      layers{l}.padding = [0 0];
    end;
    
    fan_in = outputmaps * layers{l}.kernelsize(1) *  layers{l}.kernelsize(2);
    fan_out = layers{l}.outputmaps * layers{l}.kernelsize(1) * layers{l}.kernelsize(2);
    rand_coef = 2 * sqrt(6 / (fan_in + fan_out));
    layers{l}.k = double(zeros([layers{l}.kernelsize outputmaps, layers{l}.outputmaps]));
    layers{l}.dk = double(zeros([layers{l}.kernelsize outputmaps, layers{l}.outputmaps]));
    layers{l}.dkp = double(zeros([layers{l}.kernelsize outputmaps, layers{l}.outputmaps]));
    layers{l}.gk = double(ones([layers{l}.kernelsize outputmaps, layers{l}.outputmaps]));
    layers{l}.k = (rand([layers{l}.kernelsize outputmaps layers{l}.outputmaps]) - 0.5) * rand_coef;
    layers{l}.b = double(zeros(layers{l}.outputmaps, 1));
    layers{l}.db = double(zeros(layers{l}.outputmaps, 1));
    layers{l}.dbp = double(zeros(layers{l}.outputmaps, 1));
    layers{l}.gb = double(ones(layers{l}.outputmaps, 1));
    mapsize = mapsize + 2*layers{l}.padding - layers{l}.kernelsize + 1;
    outputmaps = layers{l}.outputmaps;

  elseif strcmp(layers{l}.type, 'f') % fully connected
    if (~isfield(layers{l}, 'dropout'))
      layers{l}.dropout = 0; % no dropout
    end;
    if (~isfield(layers{l}, 'function'))
      layers{l}.function = 'sigmoid';
    end;
    if strcmp(layers{l}.function, 'SVM')
      assert(isfield(layers{l}, 'C'), 'The "SVM" layer must contain the "C" field');      
    elseif ~strcmp(layers{l}.function, 'relu') && ...
           ~strcmp(layers{l}.function, 'sigmoid')
      error('"%s" - unknown function for the layer %d', layers{l}.function, l);
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
    layers{l}.w = double((rand(weightsize) - 0.5) * 2 * sqrt(6/sum(weightsize)));
    layers{l}.dw = double(zeros(weightsize));
    layers{l}.dwp = double(zeros(weightsize));
    layers{l}.gw = double(ones(weightsize));

    layers{l}.b = double(zeros(1, weightsize(1)));
    layers{l}.db = double(zeros(1, weightsize(1)));
    layers{l}.dbp = double(zeros(1, weightsize(1)));
    layers{l}.gb = double(ones(1, weightsize(1)));      
    mapsize = [0 0];
    outputmaps = 0;      
  else
    error('"%s" - unknown type of the layer %d', layers{l}.type, l);
  end
  layers{l}.outputmaps = outputmaps;
  layers{l}.mapsize = mapsize;
end
assert(strcmp(layers{n}.type, 'f'), 'The last layer must be the type of "f"'); 
  
end
