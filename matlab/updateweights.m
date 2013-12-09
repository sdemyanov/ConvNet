function layers = updateweights(layers, params, regime)

for l = 2 : numel(layers)
  
  if strcmp(layers{l}.type, 'c')
    for i = 1 : layers{l}.outputmaps
      for j = 1 : layers{l-1}.outputmaps
        if (regime == 0)  
          layers{l}.dk{i, j} = params.momentum * layers{l}.dkp{i, j};          
        else
          if (params.adjustrate > 0)
            signs = layers{l}.dk{i, j} .* layers{l}.dkp{i, j};
            layers{l}.gk{i, j}(signs > 0) = layers{l}.gk{i, j}(signs > 0) + params.adjustrate;
            layers{l}.gk{i, j}(signs <= 0) = layers{l}.gk{i, j}(signs <= 0) * (1 - params.adjustrate);
            layers{l}.gk{i, j}(layers{l}.gk{i, j} > params.maxcoef) = params.maxcoef;
            layers{l}.gk{i, j}(layers{l}.gk{i, j} <= params.mincoef) = params.mincoef;
          end
          layers{l}.dkp{i, j} = layers{l}.dk{i, j};          
          layers{l}.dk{i, j} = (1 - params.momentum) * layers{l}.dk{i, j};
        end;
      end             
    end
    layers{l}.k = cellfun(@(k,gk,dk) k - params.alpha * gk .* dk, layers{l}.k, layers{l}.gk, layers{l}.dk, 'UniformOutput', false);
    
  elseif strcmp(layers{l}.type, 'f')
    if (regime == 0)      
      layers{l}.dw = params.momentum * layers{l}.dwp;      
    else
      if (params.adjustrate > 0)
        signs = layers{l}.dw .* layers{l}.dwp;
        layers{l}.gw(signs > 0) = layers{l}.gw(signs > 0) + params.adjustrate;
        layers{l}.gw(signs <= 0) = layers{l}.gw(signs <= 0) * (1 - params.adjustrate);
        layers{l}.gw(layers{l}.gw > params.maxcoef) = params.maxcoef;
        layers{l}.gw(layers{l}.gw <= params.mincoef) = 1/params.mincoef;  
      end;
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
      if (params.adjustrate > 0)
        signs = layers{l}.db .* layers{l}.dbp;      
        layers{l}.gb(signs > 0) = layers{l}.gb(signs > 0) + params.adjustrate;
        layers{l}.gb(signs <= 0) = layers{l}.gb(signs <= 0) * (1 - params.adjustrate);
        layers{l}.gb(layers{l}.gb > params.maxcoef) = params.maxcoef;
        layers{l}.gb(layers{l}.gb <= params.mincoef) = 1/params.mincoef;              
      end;
      layers{l}.dbp = layers{l}.db;
      layers{l}.db = (1 - params.momentum) * layers{l}.db;      
    end;        
    layers{l}.b = layers{l}.b - params.alpha * layers{l}.gb .* layers{l}.db;
  end;
end
    
end
