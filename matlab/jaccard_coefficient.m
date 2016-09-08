function [jaccardIdx,jaccardDist] = jaccard_coefficient(img_Orig,img_Seg)
% Jaccard index and distance co-efficient of segmemted and ground truth
% image
% Usage: [index,distance(JC)] = jaccard_coefficient(Orig_Image,Seg_Image);

% Check for logical image
if ~islogical(img_Orig)
    error('Image must be in logical format');
end
if ~islogical(img_Seg)
    error('Image must be in logical format');
end

% Find the intersection of the two images
inter_image = img_Orig & img_Seg;

% Find the union of the two images
union_image = img_Orig | img_Seg;

jaccardIdx = sum(inter_image(:))/sum(union_image(:));
%jaccardIdx = sum(inter_image(:));
% Jaccard distance = 1 - jaccardindex;
jaccardDist = 1 - jaccardIdx;
