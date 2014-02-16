function d = softder(d, a)

comsum = sum(a .* d, 2);
d = a .* bsxfun(@minus, d, comsum);

end

