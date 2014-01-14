function A = shrink(A, sc, st)

for d = 1 : length(size(A))
  ind = [];
  for i = 1 : sc(d) - st(d)
    ind = [ind; sc(d)+i : sc(d) : size(A, d)];
  end;
  newind = ind-sc(d)+st(d);
  if (d == 1)
    A(newind, :, :, :) = A(newind, :, :, :) + A(ind, :, :, :);
    A(ind, :, :, :) = [];
  elseif (d == 2)
    A(:, newind, :, :) = A(:, newind, :, :) + A(:, ind, :, :);
    A(:, ind, :, :) = [];
  elseif (d == 3)
    A(:, :, newind, :) = A(:, :, newind, :) + A(:, :, ind, :);
    A(:, :, ind, :) = [];
  elseif (d == 4)
    A(:, :, :, newind) = A(:, :, :, newind) + A(:, :, :, ind);
    A(:, :, :, ind) = [];
  end;  
end;

end


