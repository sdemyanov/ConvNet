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
  params.momentum = 0; % no momentum
end;
if (~isfield(params, 'adjustrate'))
  params.adjustrate = 0.05;
end;
if (~isfield(params, 'maxcoef'))
  params.maxcoef = 10;
end;

end

