kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\Results_cc100';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\Results_ext1000';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\Results_ff1000';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\Results_ff10000';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\Results_ff1000ep';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\Results_ff60000_1024';
kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\Results_cc200';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST\cc';
%kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\CIFAR';

Iter = 0;
kMaxIter = 20;

clear Beta;
n = 1;
Beta{n} = 'b'; n = n + 1;
Beta{n} = '0'; n = n + 1;
 
Beta{n} = '0.05'; n = n + 1;
Beta{n} = '0.1'; n = n + 1;
Beta{n} = '0.2'; n = n + 1; 
%Beta{n} = '0.3'; n = n + 1;
BetaNum = length(Beta);

clear LineSpec;
LineSpec{1} = 'b-+';
LineSpec{2} = 'g-o';
LineSpec{3} = 'r-*';
LineSpec{4} = 'm-x';
LineSpec{5} = 'c-s';
LineSpec{6} = 'k-d';
LineSpec{7} = 'y-^';

legstr = cell(length(Beta), 1);
for i = 1 : BetaNum
  legstr{i} = ['\beta = ' Beta{i}];
end;

%%
close all;
resultsfile = ['results-' Beta{1} '.mat'];  
load(fullfile(kWorkspaceFolder, resultsfile), 'errors');
EpochNum = 200; %size(errors, 1);
IterNum = size(errors, 2);
errmat = zeros(EpochNum, BetaNum, IterNum);
h = figure;
hold on;
for i = 1 : BetaNum
  resultsfile = ['results-' Beta{i} '.mat'];  
  load(fullfile(kWorkspaceFolder, resultsfile), 'errors');
  errmat(:, i, :) = errors(1:EpochNum, :, :) * 100;  
  if (Iter == 0)
    plot(mean(errmat(:, i, [1:8 10:kMaxIter]), 3), LineSpec{i});  
  else
    plot(errmat(:, i, Iter), LineSpec{i});  
  end;
end;
hold off;
title('Beta comparison');
xlabel('Epoch number');
ylabel('Error, %');
legend(legstr, 'Location', 'NorthEast');
axis([50 EpochNum 18 25]);
%axis([30 EpochNum 11 13]);
%axis([30 EpochNum 5.65 6.6]);
%axis([30 EpochNum 8 10]);
%axis([10 EpochNum 0.017 0.025]);
%axis([30 EpochNum 0.0565 0.066]);
%axis([30 EpochNum 4.7 5.6]);
%set(h, 'position', [700 500 380 300]);
%{
%%
h = figure;
resultsfile = ['results-' Beta{1} '.mat'];  
load(fullfile(kWorkspaceFolder, resultsfile), 'losses');
EpochNum = size(losses, 1);
IterNum = size(losses, 2);
lossmat = zeros(EpochNum, BetaNum, IterNum);
hold on;
for i = 1 : BetaNum
  resultsfile = ['results-' Beta{i} '.mat'];  
  load(fullfile(kWorkspaceFolder, resultsfile), 'losses');
  lossmat(:, i, :) = losses;
  if (Iter == 0)
    plot(mean(lossmat(:, i, 1:kMaxIter), 3), LineSpec{i});  
  else
    plot(lossmat(:, i, Iter), LineSpec{i});  
  end;
end;
hold off;
title('Loss function');
xlabel('Epoch number');
ylabel('Loss L');
legend(legstr, 'Location', 'Best');
axis([30 EpochNum 0.038 0.08]);
set(h, 'position', [700 500 380 300]);
%}
%%
%{
h = figure;
resultsfile = ['results-' Beta{1} '.mat'];  
load(fullfile(kWorkspaceFolder, resultsfile), 'losses2');
EpochNum = size(losses2, 1);
IterNum = size(losses2, 2);
lossmat2 = zeros(EpochNum, BetaNum, IterNum);
hold on;
for i = 1 : BetaNum
  resultsfile = ['results-' Beta{i} '.mat'];
  if (~strcmp(Beta{i}, 'b') && ~strcmp(Beta{i}, '0'))
    load(fullfile(kWorkspaceFolder, resultsfile), 'losses2');
    lossmat2(:, i, :) = losses2;
  else
    lossmat2(:, i, :) = 0;
  end;
  if (Iter == 0)
    plot(mean(lossmat2(:, i, 1:kMaxIter), 3), LineSpec{i+1});  
  else
    plot(lossmat2(:, i, Iter), LineSpec{i+1});  
  end;
end;
hold off;
title('Loss L_2 function');
xlabel('Epoch number');
ylabel('Loss L_2');
legend(legstr, 'Location', 'Best');
axis([1 EpochNum 0.025 0.061]);
set(h, 'position', [700 500 380 300]);
%}
