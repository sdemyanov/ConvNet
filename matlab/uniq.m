function mm = uniq(mm, sc)

shift = prod(sc);
shiftmat = reshape(shift-1 : -1 : 0, [sc(2) sc(1)])' / shift;
mmsize = [size(mm, 1) size(mm, 2) size(mm, 3) size(mm, 4)];
coefs = mmsize ./ sc;
shiftmat = repmat(shiftmat, coefs);
mm = mm + shiftmat;

fi = ceil((sc+1)/2);    
b = strel('rectangle', [sc(1) sc(2)]);
z = imdilate(mm, b);
maxval = z(fi(1):sc(1):end, fi(2):sc(2):end, :, :);
maxval = expand(maxval, sc);
mm = (mm == maxval);

end

