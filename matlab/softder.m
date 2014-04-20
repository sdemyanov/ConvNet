function d = softder(d, a)

%comsum = sum(a, 2);
%d = d .* bsxfun(@times, a, 1 - comsum);
comsum = sum(a .* d, 2);
d = a .* bsxfun(@minus, d, comsum);

end

