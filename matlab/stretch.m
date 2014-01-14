function B = stretch(A, sc, st)

dim = length(size(A));
s = ceil(size(A) ./ st);
newsize = sc .* s;
B = zeros(newsize);
order = cell(dim, 1);
ind = cell(dim, 1);
oldind = cell(dim, 1);
for d = 1 : dim
  ind{d} = [];
  for i = 1 : sc(d) - st(d)
    ind{d} = [ind{d}; sc(d)+i : sc(d) : newsize(d)];
    oldind{d} = ind{d} - sc(d) + st(d);
  end;
  order{d} = 1 : newsize(d);
  order{d}(ind{d}) = [];
  order{d}(size(A, d)+1:end) = [];
end;
B(order{:}) = A;

for d = 1 : dim
  if (d == 1)
    B(ind{d}, :, :, :) = B(oldind{d}, :, :, :);
  elseif (d == 2)
    B(:, ind{d}, :, :) = B(:, oldind{d}, :, :);
  elseif (d == 3)
    B(:, :, ind{d}, :) = B(:, :, oldind{d}, :);
  elseif (d == 4)
    B(:, :, :, ind{d}) = B(:, :, :, oldind{d});
  end;    
end;

end