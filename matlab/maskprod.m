function c = maskprod(a, a_tr, b, b_tr, m)

if (a_tr == 0 && b_tr == 0) 
  c = zeros(size(a, 1), size(b, 2));  
elseif (a_tr == 0 && b_tr > 0) 
  c = zeros(size(a, 1), size(b, 1));
  b = b';
  m = permute(m, [2 1 3]);  
elseif (a_tr > 0 && b_tr == 0) 
  c = zeros(size(a, 2), size(b, 2));
  a = a';
  m = permute(m, [3 2 1]);
elseif (a_tr > 0 && b_tr > 0) 
  assert('wrong parameters');
end;

for i = 1 : size(a, 1)
  c(i, :) = c(i, :) + a(i, :) * (b .* m(:, :, i));
end;

end

