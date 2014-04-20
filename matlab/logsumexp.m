function lcs = logsumexp(X, varargin)
% LOGSUMEXP(X, dim) computes log(sum(exp(X), dim)) robustly. Care lightspeed users!
%
%     lse = logsumexp(X[, dim]);
%
% This routine works with general ND-arrays and matches Matlab's default
% behavior for sum: if dim is omitted it sums over the first non-singleton
% dimension.
%
% Note: Tom Minka's lightspeed has a logsumexp function, which:
%     1) sets dim=1 if dim is missing
%     2) returns Inf for sums containing Infs and NaNs;
%
% This routine is fairly fast and accurate for many uses, including when all the
% values of X are large in magnitude. There is a corner case where the relative
% error is avoidably bad (although the absolute error is small), when the
% largest argument is very close to zero and the next largest is moderately
% negative. For example:
%     logsumexp([0 -40])
% Cases like this rarely come up in my work. My LOGPLUSEXP and LOGCUMSUM
% functions do cover this case.
%
% SEE ALSO: LOGCUMSUMEXP LOGPLUSEXP

% Iain Murray, September 2010

% History: IM wrote a bad logsumexp in ~2002, then used Tom Minka's version for
% years until eventually wanting something slightly different.

if (numel(varargin) > 1)
    error('Too many arguments')
end

if isempty(X)
    % Easiest way to get this trivial but annoying case right!
    lcs = log(sum(exp(X),varargin{:}));
    return;
end

if isempty(varargin)
    mx = max(X);
else
    mx = max(X, [], varargin{:});
end
Xshift = bsxfun(@minus, X, mx);
lcs = bsxfun(@plus, log(sum(exp(Xshift),varargin{:})), mx);

idx = isinf(mx);
lcs(idx) = mx(idx);
lcs(any(isnan(X),varargin{:})) = NaN;
