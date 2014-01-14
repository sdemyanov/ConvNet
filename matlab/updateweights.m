function layers = updateweights(layers, params, regime)

for l = 2 : numel(layers)
  
  if strcmp(layers{l}.type, 'c')
    if (regime == 0)  
      layers{l}.dk = params.momentum * layers{l}.dkp;          
    else
      signs = layers{l}.dk .* layers{l}.dkp;          
      layers{l}.gk(signs > 0) = layers{l}.gk(signs > 0) + params.adjustrate;
      layers{l}.gk(signs <= 0) = layers{l}.gk(signs <= 0) * (1 - params.adjustrate);
      layers{l}.gk(layers{l}.gk > params.maxcoef) = params.maxcoef;
      layers{l}.gk(layers{l}.gk <= params.mincoef) = params.mincoef;          
      layers{l}.dkp = layers{l}.dk;          
      layers{l}.dk = (1 - params.momentum) * layers{l}.dk;
    end;
    layers{l}.k = layers{l}.k - params.alpha * layers{l}.gk .* layers{l}.dk;                      
    
  elseif strcmp(layers{l}.type, 'f')
    if (regime == 0)      
      layers{l}.dw = params.momentum * layers{l}.dwp;      
    else
      signs = layers{l}.dw .* layers{l}.dwp;
      layers{l}.gw(signs > 0) = layers{l}.gw(signs > 0) + params.adjustrate;
      layers{l}.gw(signs <= 0) = layers{l}.gw(signs <= 0) * (1 - params.adjustrate);
      layers{l}.gw(layers{l}.gw > params.maxcoef) = params.maxcoef;
      layers{l}.gw(layers{l}.gw <= params.mincoef) = 1/params.mincoef;  
      layers{l}.dwp = layers{l}.dw;
      layers{l}.dw = (1 - params.momentum) * layers{l}.dw;      
    end;  
    layers{l}.w = layers{l}.w - params.alpha * layers{l}.gw .* layers{l}.dw;
    %constr = 0.4;
    %layers{l}.w(layers{l}.w > constr) = constr;
    %layers{l}.w(layers{l}.w < -constr) = -constr;  
  end
  
  % for all transforming layers
  if strcmp(layers{l}.type, 'c') || strcmp(layers{l}.type, 'f')
    if (regime == 0)  
      layers{l}.db = params.momentum * layers{l}.dbp;      
    else      
      signs = layers{l}.db .* layers{l}.dbp;      
      layers{l}.gb(signs > 0) = layers{l}.gb(signs > 0) + params.adjustrate;
      layers{l}.gb(signs <= 0) = layers{l}.gb(signs <= 0) * (1 - params.adjustrate);
      layers{l}.gb(layers{l}.gb > params.maxcoef) = params.maxcoef;
      layers{l}.gb(layers{l}.gb <= params.mincoef) = 1/params.mincoef;              
      layers{l}.dbp = layers{l}.db;
      layers{l}.db = (1 - params.momentum) * layers{l}.db;      
    end;        
    layers{l}.b = layers{l}.b - params.alpha * layers{l}.gb .* layers{l}.db;
  end;
end
    
end
