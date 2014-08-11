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
  params.momentum = 0;
end;
if (~isfield(params, 'adjustrate'))
  params.adjustrate = 0;
end;
if (~isfield(params, 'maxcoef'))
  params.maxcoef = 1;
  params.mincoef = 1;
else
  params.mincoef = 1 / params.maxcoef;
end;
if (~isfield(params, 'shuffle'))
  params.shuffle = 1;
end;
if (~isfield(params, 'verbose'))
  params.verbose = 2;
end;
if (~isfield(params, 'seed'))
  params.seed = 0;
end;

end

