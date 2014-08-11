function maxmat = uniq(maxmat, sc)

shift = prod(sc);
shiftmat = reshape(shift-1 : -1 : 0, [sc(2) sc(1)])' / shift;
coefs = size(maxmat) ./ sc;
shiftmat = repmat(shiftmat, coefs);
maxmat = maxmat + shiftmat;

fi = ceil((sc+1)/2);    
b = strel('rectangle', [sc(1) sc(2)]);
z = imdilate(maxmat, b);
maxval = z(fi(1):sc(1):end, fi(2):sc(2):end, :, :);
maxval = expand(maxval, sc);
maxmat = (maxmat == maxval);

end

