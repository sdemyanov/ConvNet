function layers = updateweights(layers, params, epoch, regime)

for l = 1 : numel(layers)
  
  if (length(params.momentum) == 1)
    momentum = params.momentum;
  else
    momentum = params.momentum(epoch);
  end;
  if (length(params.alpha) == 1)
    alpha = params.alpha;
  else
    alpha = params.alpha(epoch);
  end;
  
  if strcmp(layers{l}.type, 'c')    
    if (regime == 0)  
      dk = momentum * layers{l}.dkp;          
    else
      dk = alpha * layers{l}.dk;
      signs = dk .* layers{l}.dkp;          
      layers{l}.gk(signs > 0) = layers{l}.gk(signs > 0) + params.adjustrate;
      layers{l}.gk(signs <= 0) = layers{l}.gk(signs <= 0) * (1 - params.adjustrate);
      layers{l}.gk(layers{l}.gk > params.maxcoef) = params.maxcoef;
      layers{l}.gk(layers{l}.gk <= params.mincoef) = params.mincoef;          
      dk = dk .* layers{l}.gk;
      layers{l}.dkp = dk;      
      dk = (1 - momentum) * dk;
    end;
    layers{l}.k = layers{l}.k - dk;                      
    
  elseif strcmp(layers{l}.type, 'f')
    if (regime == 0)      
      dw = momentum * layers{l}.dwp;      
    else
      dw = alpha * layers{l}.dw;
      signs = layers{l}.dw .* layers{l}.dwp;
      layers{l}.gw(signs > 0) = layers{l}.gw(signs > 0) + params.adjustrate;
      layers{l}.gw(signs <= 0) = layers{l}.gw(signs <= 0) * (1 - params.adjustrate);
      layers{l}.gw(layers{l}.gw > params.maxcoef) = params.maxcoef;
      layers{l}.gw(layers{l}.gw <= params.mincoef) = 1/params.mincoef;  
      dw = dw .* layers{l}.gw;
      layers{l}.dwp = dw;
      dw = (1 - momentum) * dw;      
    end;  
    layers{l}.w = layers{l}.w - dw;    
  end
  
  % for all transforming layers
  if strcmp(layers{l}.type, 'c') || strcmp(layers{l}.type, 'f')
    if (regime == 0)  
      db = momentum * layers{l}.dbp;      
    else
      db = alpha * layers{l}.db;
      signs = db .* layers{l}.dbp;      
      layers{l}.gb(signs > 0) = layers{l}.gb(signs > 0) + params.adjustrate;
      layers{l}.gb(signs <= 0) = layers{l}.gb(signs <= 0) * (1 - params.adjustrate);
      layers{l}.gb(layers{l}.gb > params.maxcoef) = params.maxcoef;
      layers{l}.gb(layers{l}.gb <= params.mincoef) = 1/params.mincoef;              
      db = db .* layers{l}.gb;
      layers{l}.dbp = db;
      db = (1 - momentum) * db;      
    end;        
    layers{l}.b = layers{l}.b - db;
  end;
end
    
end
