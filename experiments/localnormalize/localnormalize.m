function ln=localnormalize(IM,sigma1,sigma2)
%LOCALNORMALIZE A local normalization algorithm that uniformizes the local
%mean and variance of an image.
%  ln=localnormalize(IM,sigma1,sigma2) outputs local normalization effect of 
%  image IM using local mean and standard deviation estimated by Gaussian
%  kernel with sigma1 and sigma2 respectively.
%
%  Contributed by Guanglei Xiong (xgl99@mails.tsinghua.edu.cn)
%  at Tsinghua University, Beijing, China.
epsilon=1e-1;
halfsize1=ceil(-norminv(epsilon/2,0,sigma1));
size1=2*halfsize1+1;
halfsize2=ceil(-norminv(epsilon/2,0,sigma2));
size2=2*halfsize2+1;
gaussian1=fspecial('gaussian',size1,sigma1);
gaussian2=fspecial('gaussian',size2,sigma2);
num=IM-imfilter(IM,gaussian1);
den=sqrt(imfilter(num.^2,gaussian2));
ln=num./den;