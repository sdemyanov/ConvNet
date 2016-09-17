function layers = setup(layers)
    
assert(strcmp(layers{1}.type, 'input'), 'The first layer must be the type of "input"');
n = numel(layers);
for l = 1 : n   %  layer
  
  if strcmp(layers{l}.type, 'input') % input
    assert(isfield(layers{l}, 'mapsize'), 'The "input" type layer must contain the "mapsize" field');
    assert(numel(layers{l}.mapsize) == 2, 'The input layer mapsize must have the length 2');
    if (~isfield(layers{l}, 'channels'))
      layers{l}.channels = 1;
    end;
    if (~isfield(layers{l}, 'add_bias'))
      layers{l}.add_bias = 0;
    end;
    
  
  elseif strcmp(layers{l}.type, 'jitt') % jittering
    assert(isfield(layers{l}, 'mapsize'), 'The "jitt" type layer must contain the "mapsize" field');    
    layers{l}.channels = layers{l-1}.channels;
    if (~isfield(layers{l}, 'add_bias'))
      layers{l}.add_bias = 0;
    end;
    
  
  elseif strcmp(layers{l}.type, 'pool') % scaling
    assert(isfield(layers{l}, 'scale'), 'The "pool" type layer must contain the "scale" field');
    assert(isfield(layers{l}, 'stride'), 'The "pool" type layer must contain the "stride" field');
    if (~isfield(layers{l}, 'padding'))
      layers{l}.padding = [0 0];
    end;
    layers{l}.mapsize = 1 + floor((layers{l-1}.mapsize +2*layers{l}.padding - layers{l}.scale) ./ layers{l}.stride);
    layers{l}.channels = layers{l-1}.channels;
    if (~isfield(layers{l}, 'add_bias'))
      layers{l}.add_bias = 0;
    end;
    
    
  elseif strcmp(layers{l}.type, 'conv') % convolutional
    assert(isfield(layers{l}, 'filtersize'), 'The "conv" type layer must contain the "filtersize" field');
    assert(isfield(layers{l}, 'channels'), 'The "conv" type layer must contain the "channels" field');
    if (~isfield(layers{l}, 'padding'))
      layers{l}.padding = [0 0];
    end;
    if (~isfield(layers{l}, 'stride'))
      layers{l}.stride = [1 1];
    end;
    layers{l}.mapsize = 1 + floor(layers{l-1}.mapsize + 2*layers{l}.padding - layers{l}.filtersize) ./ layers{l}.stride;
    layers{l}.weightsize = [layers{l}.filtersize(2) layers{l}.filtersize(1) layers{l-1}.channels layers{l}.channels];    
    
    
  elseif strcmp(layers{l}.type, 'deconv') % deconvolutional
    assert(isfield(layers{l}, 'filtersize'), 'The "deconv" type layer must contain the "filtersize" field');
    assert(isfield(layers{l}, 'channels'), 'The "deconv" type layer must contain the "channels" field');
    if (~isfield(layers{l}, 'stride'))
      layers{l}.stride = [1 1];
    end;
    layers{l}.mapsize = (layers{l-1}.mapsize - 1) .* layers{l}.stride + layers{l}.filtersize;
    layers{l}.weightsize = [layers{l}.filtersize(2) layers{l}.filtersize(1) layers{l}.channels layers{l-1}.channels];    
    

  elseif strcmp(layers{l}.type, 'full') % fully connected
    assert(isfield(layers{l}, 'channels'), 'The "full" type layer must contain the "channels" field');
    layers{l}.mapsize = [1 1];
    layers{l}.weightsize = [prod(layers{l-1}.mapsize) * layers{l-1}.channels layers{l}.channels];    
    
    
  else
    error('"%s" - unknown type of the layer %d', layers{l}.type, l);
  end
  
  if (~isfield(layers{l}, 'add_bias') || layers{l}.add_bias > 0)
    layers{l}.biassize = [layers{l}.channels 1];
  end;
  
end
  
end
