function layers = cnnsetup(layers)
    
assert(strcmp(layers{1}.type, 'i'), 'The first layer must be the type of "i"');
assert(isfield(layers{1}, 'mapsize'), 'The first layer must contain the "mapsize" field');
mapsize = layers{1}.mapsize;
if (~isfield(layers{1}, 'outputmaps'))
  outputmaps = 1;
else
  outputmaps = layers{1}.outputmaps;
end;    
n = numel(layers);
for l = 1 : n   %  layer
  if strcmp(layers{l}.type, 's') % scaling
    assert(isfield(layers{l}, 'scale'), 'The "s" type layer must contain the "scale" field');
    if (~isfield(layers{l}, 'function'))
      layers{l}.function = 'max';
    end;
    if ~strcmp(layers{l}.function, 'max') && ...
       ~strcmp(layers{l}.function, 'mean')
      error('"%s" - unknown function for the layer %d', layers{l}.function, l);
    end;
    mapsize = ceil(mapsize ./ layers{l}.scale);
    
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
    
    fan_in = outputmaps * layers{l}.kernelsize(1) *  layers{l}.kernelsize(2);
    fan_out = layers{l}.outputmaps * layers{l}.kernelsize(1) * layers{l}.kernelsize(2);
    rand_coef = 2 * sqrt(6 / (fan_in + fan_out));
    for i = 1 : layers{l}.outputmaps  %  output map
      for j = 1 : outputmaps  %  input map
        layers{l}.k{i, j} = (rand(layers{l}.kernelsize(1), layers{l}.kernelsize(2)) - 0.5) * rand_coef;
        layers{l}.dk{i, j} = zeros(layers{l}.kernelsize(1), layers{l}.kernelsize(2));
        layers{l}.dkp{i, j} = zeros(layers{l}.kernelsize(1), layers{l}.kernelsize(2));
        layers{l}.gk{i, j} = ones(layers{l}.kernelsize(1), layers{l}.kernelsize(2));
      end      
    end
    layers{l}.b = zeros(layers{l}.outputmaps, 1);
    layers{l}.db = zeros(layers{l}.outputmaps, 1);
    layers{l}.dbp = zeros(layers{l}.outputmaps, 1);
    layers{l}.gb = ones(layers{l}.outputmaps, 1);
    mapsize = mapsize - layers{l}.kernelsize + 1;
    outputmaps = layers{l}.outputmaps;

  elseif strcmp(layers{l}.type, 'f') % fully connected
    if (~isfield(layers{l}, 'droprate'))
      layers{l}.droprate = 0; % no dropout
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
    layers{l}.w = (rand(weightsize) - 0.5) * 2 * sqrt(6/sum(weightsize));
    layers{l}.dw = zeros(weightsize);
    layers{l}.dwp = zeros(weightsize);
    layers{l}.gw = ones(weightsize);

    layers{l}.b = zeros(weightsize(1), 1);
    layers{l}.db = zeros(weightsize(1), 1);
    layers{l}.dbp = zeros(weightsize(1), 1);
    layers{l}.gb = ones(weightsize(1), 1);      
    mapsize = [0 0];
    outputmaps = 0;      
  elseif ~strcmp(layers{l}.type, 'i')
    error('"%s" - unknown type of the layer %d', layers{l}.type, l);
  end
  layers{l}.outputmaps = outputmaps;
  layers{l}.mapsize = mapsize;
end
assert(strcmp(layers{n}.type, 'f'), 'The last layer must be the type of "f"'); 
  
end
