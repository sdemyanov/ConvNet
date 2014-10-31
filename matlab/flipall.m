function X = flipall(X)
  for dimind = 1 : ndims(X)
    X = flip(X, dimind);
  end
end