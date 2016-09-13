% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy] = get_accuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;
end
