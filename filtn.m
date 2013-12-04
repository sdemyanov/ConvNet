function c = filtn(a, b, type)

c = convn(a, flipall(b), type);

function X = flipall(X)
  for dimind = 1 : ndims(X)
    X = flipdim(X, dimind);
  end
end

end

