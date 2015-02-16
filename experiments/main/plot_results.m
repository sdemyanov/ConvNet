%datasetName = 'MNIST';
datasetName = 'CIFAR';
%datasetName = 'CIFAR-100';
%datasetName = 'SVHN';
%datasetName = 'tangent';

if (ispc)
  kWorkspaceFolder = ['C:\Users\sergeyd\Workspaces\' datasetName];
else
  kWorkspaceFolder = ['/media/sergeyd/OS/Users/sergeyd/Workspaces/' datasetName];  
end;

Iter = 0;
kMaxIter = [1:10];

isplot = 0;
iserr = 0;
isloss = 1;
isloss2 = 1;

clear Beta;
n = 1;

if (strcmp(datasetName, 'MNIST'))    
  
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'final/10k');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'valid/50k');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'valid-new/6k');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'mnist-trans/6k');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'mnist-notrans/60k');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'CNN-final');
  
  kWorkspaceFolder = fullfile(kWorkspaceFolder, 'final-last');  
  Beta{n} = 'n-a0.1-b0'; n = n + 1;
  Beta{n} = 'n-a0.1-b0.002'; n = n + 1;
  Beta{n} = 'n-a0.1-b0-d0.5'; n = n + 1;  
  Beta{n} = 'n-a0.1-b0.0002-d0.5'; n = n + 1;
  
  %Beta{n} = 'n-a0.1-b0'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.1'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.01'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.005'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.003'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.005'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b0.01'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.02'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.03'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.05'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.1'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.03'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.3'; n = n + 1;
  
  legstr = Beta; n = 1;
  %{
  legstr{n} = '[0, 0]'; n = n + 1;
  legstr{n} = '[0, 2e-3]'; n = n + 1;
  legstr{n} = '[0.5, 0]';  n = n + 1;
  legstr{n} = '[0.5, 2e-4]'; n = n + 1;
  %}
  IterNum = 10;
  EpochNum = 400; %size(curerr, 1);
  Period = 10;
  
  %mnist_axis = [0 EpochNum 3 7];
  %mnist_axis = [0 EpochNum 2 4];
  mnist_axis = [0 EpochNum 0 2];  
  
elseif (strcmp(datasetName, 'CIFAR') || strcmp(datasetName, 'CIFAR-100'))
  
  kWorkspaceFolder = fullfile(kWorkspaceFolder, 'final-last');
  Beta{n} = 'n-m0.9-a0.1-b0'; n = n + 1;
  Beta{n} = 'n-m0.9-a0.1-b0.0005'; n = n + 1;
  Beta{n} = 'n-m0.9-a0.1-b0-d0.5'; n = n + 1;  
  Beta{n} = 'n-m0.9-a0.1-b5e-05-d0.5'; n = n + 1;
  
  %Beta{n} = 'n-a0.1-b1e-06-tang'; n = n + 1;  
  
  %Beta{n} = 'n-a0.1-b0.0001'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.001'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.01'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b0.02'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b0.03'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b0.05'; n = n + 1;  
  
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'valid-drop');
  
  %Beta{n} = 'n-m0.9-a0.1-b0-d0.5'; n = n + 1;
  %Beta{n} = 'n-m0.9-a0.1-b3e-05-d0.5'; n = n + 1;
  %Beta{n} = 'n-m0.9-a0.1-b0.0002-d0.5'; n = n + 1;
  %Beta{n} = 'n-m0.9-a0.1-b0.0003-d0.5'; n = n + 1;
  %Beta{n} = 'n-m0.9-a0.1-b0.0005-d0.5'; n = n + 1;
  %Beta{n} = 'n-m0.9-a0.1-b0.001-d0.5'; n = n + 1;
  
  legstr = Beta; n = 1;
  legstr{n} = 'BP ($\beta = 0$), no d/o';  n = n + 1;
  legstr{n} = 'IBP ($\beta$ = 5e-4), no d/o';  n = n + 1;
  legstr{n} = 'BP ($\beta = 0$), with d/o';  n = n + 1;
  legstr{n} = 'IBP ($\beta$ = 5e-5), with d/o';  n = n + 1;

  IterNum = 10;
  EpochNum = 800; %size(curerr, 1);
  Period = 40;
  
elseif (strcmp(datasetName, 'SVHN'))   
  kWorkspaceFolder = fullfile(kWorkspaceFolder, 'final-last');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'iter1');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, '2layers-val');
  
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'valid-nodrop');
  Beta{n} = 'm0.9-a0.1-b0'; n = n + 1;  
  %Beta{n} = 'm0.9-a0.1-b0.0001'; n = n + 1;  
  %Beta{n} = 'm0.9-a0.1-b0.0003'; n = n + 1;  
  %Beta{n} = 'm0.9-a0.1-b0.001'; n = n + 1;  
  %Beta{n} = 'm0.9-a0.1-b0.003'; n = n + 1;  
  Beta{n} = 'm0.9-a0.1-b0.01'; n = n + 1;  
  %Beta{n} = 'm0.9-a0.1-b0.03'; n = n + 1;  
  %Beta{n} = 'm0.9-a0.1-b0.1'; n = n + 1;    
  
  Beta{n} = 'm0.9-a0.1-b0-d0.5'; n = n + 1;  
  %Beta{n} = 'm0.9-a0.1-b1e-06-d0.5'; n = n + 1;
  %Beta{n} = 'm0.9-a0.1-b1e-05-d0.5'; n = n + 1;
  %Beta{n} = 'm0.9-a0.1-b0.0001-d0.5'; n = n + 1;
  Beta{n} = 'm0.9-a0.1-b0.001-d0.5'; n = n + 1;
  %Beta{n} = 'm0.9-a0.1-b0.01-d0.5'; n = n + 1;
  %Beta{n} = 'm0.9-a0.1-b0.1-d0.5'; n = n + 1;
  %Beta{n} = 'm0.9-a0.1-b0.0002'; n = n + 1;
  %Beta{n} = 'm0.9-a0.1-b0.0003'; n = n + 1;
  %Beta{n} = 'm0.9-a0.1-b0.0005'; n = n + 1;
  
  
  legstr = Beta;  n = 1;
  %legstr{n} = '[0, 0]'; n = n + 1;
  %legstr{n} = '[0.5, 0]'; n = n + 1;
  %legstr{n} = '[0, 2e-3]'; n = n + 1;
  %legstr{n} = '[0.5, 2e-4]'; n = n + 1;
  
  IterNum = 10;
  EpochNum = 80; %size(curerr, 1);
  Period = 1;
  
  cifar_axis = [0 EpochNum 3 4];
  %cifar_axis = [0 EpochNum 5 10];

elseif (strcmp(datasetName, 'tangent'))   
  
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'mnist10-notrans-final');
  %kWorkspaceFolder = fullfile(kWorkspaceFolder, 'cifar50');
  kWorkspaceFolder = fullfile(kWorkspaceFolder, 'mnist60');
  
  Beta{n} = 'n-a0.1-b0'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b1e-14'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-13'; n = n + 1;
  %Beta{n} = 'n-a0.1-b3e-13'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-12'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b3e-12'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-11'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b1e-10'; n = n + 1;  
  %Beta{n} = 'n-a0.1-b1e-09'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-09-w'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-08'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.001'; n = n + 1;
  %Beta{n} = 'n-a0.1-b0.0001'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-05'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-06'; n = n + 1;
  %Beta{n} = 'n-a0.1-b1e-07'; n = n + 1;
  
  Beta{n} = 'n-a0.1-b1e-07-tang'; n = n + 1;  
  Beta{n} = 'n-a0.1-b0.03-ibp'; n = n + 1;
  
  legstr = Beta; n = 1;
  
  IterNum = 10;
  EpochNum = 200;
  Period = 10;
  
end;

BetaNum = length(Beta);

clear LineSpec;
LineSpec{1} = 'b-+';
LineSpec{2} = 'go--';
LineSpec{3} = 'r:*';
LineSpec{4} = 'm-.x';
LineSpec{5} = 'c-s';
LineSpec{6} = 'k--d';
LineSpec{7} = 'y:^';
LineSpec{8} = 'b-.+';
LineSpec{9} = 'g-o';
%%

close all;
resultsfile = ['results-' Beta{1} '.mat'];  
load(fullfile(kWorkspaceFolder, resultsfile), 'curerr');
%IterNum = size(curerr, 2);
%EpochNum = size(curerr, 1);
errmat = zeros(EpochNum, IterNum, BetaNum);

err = zeros(BetaNum, 2);
if (isplot > 0)
  h = figure;
  hold on;
end;
for i = 1 : BetaNum
  resultsfile = ['results-' Beta{i} '.mat'];  
  load(fullfile(kWorkspaceFolder, resultsfile), 'curerr');
  errmat(:, :, i) = curerr(1:EpochNum, 1:IterNum, :) * 100;  
  if (Iter == 0) 
    plotmat = mean(errmat(:, kMaxIter, i), 2);    
    err(i, 1) = mean(errmat(end, kMaxIter, i));
    err(i, 2) = std(errmat(end, kMaxIter, i)); 
  else
    plotmat = errmat(:, Iter, i);    
  end;
  if (isplot > 0)
    if (Period > 1)
      plotmat = reshape(plotmat, [Period EpochNum/Period]);
      plotmat = squeeze(mean(plotmat, 1));
      plot([1 : Period : EpochNum], plotmat, LineSpec{i});
    else
      plot(plotmat, LineSpec{i});        
    end;  
  end;
end;
if (isplot > 0)
  hold off;
  %title('Test error curves');
  xlabel('Epoch number');
  ylabel('Error, %');
  legend(legstr, 'Location', 'NorthEast','Interpreter','latex');
  if (strcmp(datasetName, 'MNIST'))
    axis(mnist_axis);
  elseif (strcmp(datasetName, 'CIFAR'))
    axis([0 EpochNum 43 53]);
    %axis([0 EpochNum 15 19]);
    %axis([0 EpochNum 16 20]);
  elseif (strcmp(datasetName, 'CIFAR-100'))
    axis([0 EpochNum 44 55]);
    %axis([0 EpochNum 16 20]);
  elseif (strcmp(datasetName, 'SVHN'))
    axis(cifar_axis);    
  elseif (strcmp(datasetName, 'tangent'))
    %axis([0 EpochNum 45 53]);
    %axis([0 EpochNum 1 3]);
    %axis([0 EpochNum 3 7]);
    %axis([0 EpochNum 0.2 1]);
    %axis([0 EpochNum 30 35]);
  end;
end;

%set(h, 'position', [700 500 380 300]);
if (Iter == 0 && iserr > 0)
  h = figure;
  lastval = squeeze(errmat(EpochNum, kMaxIter, :));
  boxplot(lastval, 'labels', legstr);
  %errorbar(1:4, mean(lastval), std(lastval));
  %title('Final error ratios');
  xlabel('Dropout and \beta values');
  ylabel('Error, %');
  set(h, 'position', [700 500 380 300]);
  %{
  h = figure;
  weightsmat = zeros(BetaNum, IterNum);
  for i = 1 : BetaNum
    resultsfile = ['results-' Beta{i} '.mat'];  
    load(fullfile(kWorkspaceFolder, resultsfile), 'WeightsIn');
    weightsmat(i, :) = sum(WeightsIn.^2, 1);  
  end;
  boxplot(weightsmat', 'labels', legstr);
  xlabel('\beta');
  ylabel('\Sigma_i w_i^2');
  set(h, 'position', [700 500 380 300]);
  %}
end;

%{
if (Iter == 0)
  h = figure;
  lastval = squeeze(errmat(EpochNum, kMaxIter, :));
  diffmat = bsxfun(@rdivide, lastval(:, 2:end), lastval(:, 1));
  boxplot(diffmat, 'labels', legstr(2:end));
  %title('Final error ratios');
  xlabel('\beta');
  ylabel('E_\beta / E_0');
  set(h, 'position', [700 500 380 300]);
  %{
  h = figure;
  weightsmat = zeros(BetaNum, IterNum);
  for i = 1 : BetaNum
    resultsfile = ['results-' Beta{i} '.mat'];  
    load(fullfile(kWorkspaceFolder, resultsfile), 'WeightsIn');
    weightsmat(i, :) = sum(WeightsIn.^2, 1);  
  end;
  boxplot(weightsmat', 'labels', legstr);
  xlabel('\beta');
  ylabel('\Sigma_i w_i^2');
  set(h, 'position', [700 500 380 300]);
  %}
end;
%}


%%
if (isloss > 0)
  resultsfile = ['results-' Beta{1} '.mat'];
  errmat = zeros(EpochNum, IterNum, BetaNum);
  h = figure;
  hold on;
  for i = 1 : BetaNum
    resultsfile = ['results-' Beta{i} '.mat'];  
    load(fullfile(kWorkspaceFolder, resultsfile), 'curloss');
    errmat(:, :, i) = curloss(1:EpochNum, 1:IterNum, :);  
    if (Iter == 0) 
      plotmat = mean(errmat(:, kMaxIter, i), 2);    
    else
      plotmat = errmat(:, Iter, i);    
    end;
    if (Period > 1)
      plotmat = reshape(plotmat, [Period EpochNum/Period]);
      plotmat = squeeze(mean(plotmat, 1));
      plot([1 : Period : EpochNum], plotmat, LineSpec{i});
    else
      plot(plotmat, LineSpec{i});  
    end;  
  end;
  hold off;
  %title('Loss function');
  xlabel('Epoch number');
  ylabel('Loss L');
  legend(legstr, 'Location', 'Best','Interpreter','latex');
  axis([0 EpochNum 0.1 0.6]);
  set(h, 'position', [700 500 380 240]);
end;
%%

if (isloss2 > 0)
  resultsfile = ['results-' Beta{1} '.mat'];  
  errmat = zeros(EpochNum, IterNum, BetaNum);
  h = figure;
  hold on;
  for i = 1 : BetaNum
    resultsfile = ['results-' Beta{i} '.mat'];  
    load(fullfile(kWorkspaceFolder, resultsfile), 'curloss2');
    errmat(:, :, i) = curloss2(1:EpochNum, 1:IterNum, :);  
    if (Iter == 0) 
      plotmat = mean(errmat(:, kMaxIter, i), 2);    
    else
      plotmat = errmat(:, Iter, i);    
    end;
    if (Period > 1)
      plotmat = reshape(plotmat, [Period EpochNum/Period]);
      plotmat = squeeze(mean(plotmat, 1));
      plot([1 : Period : EpochNum], plotmat, LineSpec{i});
    else
      plot(plotmat, LineSpec{i});  
    end;  
  end;
  hold off;
  %title('Loss L_2 function');
  xlabel('Epoch number');
  %ylabel('Second loss function');
  legend(legstr, 'Location', 'Best','Interpreter','latex');
  %axis([1 EpochNum 0 0.002]);
  set(h, 'position', [700 500 380 240]);
end;
%%
%{
shiftfile = 'jittering.mat';  
load(fullfile(kWorkspaceFolder, shiftfile), 'errors');
EpochNum = size(errors, 1);
IterNum = size(errors, 3);
errmat = errors * 100;
figure;
hold on;
for i = 1 : BetaNum
  if (Iter == 0)    
    plot(mean(errmat(:, i, kMaxIter), 3), LineSpec{i});  
  else
    plot(errmat(:, i, Iter), LineSpec{i});
  end;
end;
hold off;
title('Intensity stability');
xlabel('Max brightness value, 1e-2');
ylabel('Error, %');
legend(legstr, 'Location', 'Best');
axis([1 EpochNum 9 20]);
%}
%{
EpochNum = 100;
BetaNum = 6;
IterNum  = 12;
errors = zeros(EpochNum, BetaNum, IterNum);

shiftfile = ['shift-' num2str(1) '.mat'];  
load(fullfile(kWorkspaceFolder, shiftfile), 'errors_shift');
errors(:,:,1:3) = errors_shift(:,:,1:3);
shiftfile = ['shift-' num2str(6) '.mat'];  
load(fullfile(kWorkspaceFolder, shiftfile), 'errors_shift');
errors(:,:,4:6) = errors_shift(:,:,6:8);
shiftfile = ['shift-' num2str(11) '.mat'];  
load(fullfile(kWorkspaceFolder, shiftfile), 'errors_shift');
errors(:,:,7:9) = errors_shift(:,:,11:13);
shiftfile = ['shift-' num2str(16) '.mat'];  
load(fullfile(kWorkspaceFolder, shiftfile), 'errors_shift');
errors(:,:,10:12) = errors_shift(:,:,16:18);
errors(93:end, :, :) = [];
%%
close all;
figure; hold on;
errmat = mean(errors, 3);
for i = 1 : BetaNum
  plot(errmat(:,i,:), LineSpec{i});
end;
legend(legstr, 'Location', 'Best');
axis([1 30 0.01 0.1]);  
%}