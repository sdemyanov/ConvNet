function [params] = setparams(params)

if (~isfield(params, 'batchsize'))
  params.batchsize = 50;  
end;
if (~isfield(params, 'numepochs'))
  params.numepochs = 1;  
end;  
if (~isfield(params, 'balance'))
  params.balance = 0;
end;
if (~isfield(params, 'alpha'))
  params.alpha = 1;  
end;
if (~isfield(params, 'momentum'))
  params.momentum = 0.5;
end;
if (~isfield(params, 'adjustrate'))
  params.adjustrate = 0;
end;
if (~isfield(params, 'maxcoef'))
  params.maxcoef = 10;
  params.mincoef = 0.1;
else
  params.mincoef = 1 / params.maxcoef;
end;
if (~isfield(params, 'verbose'))
  params.verbose = 2;
end;

end

