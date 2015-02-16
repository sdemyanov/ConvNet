clear;close all;
im=imread('rice.png');
fim=mat2gray(im);
lnfim=localnormalize(fim,4,4);
lnfim=mat2gray(lnfim);
imshow(fim);
figure,imshow(lnfim);