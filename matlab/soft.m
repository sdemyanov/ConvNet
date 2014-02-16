function a = soft(a)

a = exp(bsxfun(@minus, a, logsumexp(a, 2)));
%a = exp(a) ./ repmat(sum(exp(a), 2), 1, size(a, 2));

end

