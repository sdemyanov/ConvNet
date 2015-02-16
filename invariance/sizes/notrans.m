close all;

datasetName = 'MNIST';
%datasetName = 'CIFAR';
%datasetName = 'CIFAR-100';
%datasetName = 'SVHN';

if (ispc)
  kBaseFolder = ['C:\Users\sergeyd\Workspaces\' datasetName];
else
  kBaseFolder = ['/media/sergeyd/OS/Users/sergeyd/Workspaces/' datasetName];  
end;

kWorkspaceFolder = fullfile(kBaseFolder, 'mnist-notrans');

Orig = 'n-a0.1-b0';
clear Beta; n = 1;
Beta{n} = 'n-a0.1-b0.2'; n = n + 1;
Beta{n} = 'n-a0.1-b0.15'; n = n + 1;
Beta{n} = 'n-a0.1-b0.1'; n = n + 1;
Beta{n} = 'n-a0.1-b0.05'; n = n + 1;
Beta{n} = 'n-a0.1-b0.03'; n = n + 1;
Beta{n} = 'n-a0.1-b0.02'; n = n + 1;
Beta{n} = 'n-a0.1-b0.02'; n = n + 1;
Beta{n} = 'n-a0.1-b0.02'; n = n + 1;

BetaNum = length(Beta);

clear legstr;
legstr{1} = '[0, 0]';
legstr{2} = '[0, 2e-3]';

clear subfolder;
subfolder{1} = '1k';
subfolder{2} = '2k';
subfolder{3} = '3k';
subfolder{4} = '6k';
subfolder{5} = '10k';
subfolder{6} = '20k';
subfolder{7} = '30k';
subfolder{8} = '60k';

SizesNum = length(subfolder);

IterNum = 5; %size(curerr, 2);
EpochNum = 150; %size(curerr, 1);
errmat = zeros(IterNum, SizesNum, 2);

for i = 1 : SizesNum
  curfolder = fullfile(kWorkspaceFolder, subfolder{i});
  resultsfile = ['results-' Orig '.mat'];  
  load(fullfile(curfolder, resultsfile), 'curerr');
  errmat(:, i, 1) = curerr(end, 1:IterNum);  
  resultsfile = ['results-' Beta{i} '.mat'];  
  load(fullfile(curfolder, resultsfile), 'curerr');
  errmat(:, i, 2) = curerr(end, 1:IterNum);    
end; 

LineSpec{1} = 'b-o';
xvalues = [1 2 3 6 10 20 30 60];
val1 = mean(errmat(:, :, 1), 1) * 100;
val2 = mean(errmat(:, :, 2), 1) * 100;

%{ ---------------------------------------------------- %}

%kWorkspaceFolder = fullfile(kBaseFolder, 'mnist-trans');
kWorkspaceFolder = fullfile(kBaseFolder, 'valid-new');

Orig = 'n-a0.1-b0';

clear Beta; n = 1;
Beta{n} = 'n-a0.1-b0.01'; n = n + 1;
Beta{n} = 'n-a0.1-b0.005'; n = n + 1;
Beta{n} = 'n-a0.1-b0.003'; n = n + 1;
Beta{n} = 'n-a0.1-b0.003'; n = n + 1;
Beta{n} = 'n-a0.1-b0.002'; n = n + 1;
Beta{n} = 'n-a0.1-b0.002'; n = n + 1;
Beta{n} = 'n-a0.1-b0.002'; n = n + 1;
Beta{n} = 'n-a0.1-b0.002'; n = n + 1;

BetaNum = length(Beta);

clear legstr;
legstr{1} = '[0, 0]';
legstr{2} = '[0, 2e-3]';

clear subfolder;
subfolder{1} = '1k';
subfolder{2} = '2k';
subfolder{3} = '3k';
subfolder{4} = '5k';
subfolder{5} = '10k';
subfolder{6} = '20k';
subfolder{7} = '30k';
subfolder{8} = '60k';

SizesNum = length(subfolder);

IterNum = 5; %size(curerr, 2);
EpochNum = 250; %size(curerr, 1);
errmat = zeros(IterNum, SizesNum, 2);

for i = 1 : SizesNum
  curfolder = fullfile(kWorkspaceFolder, subfolder{i});
  resultsfile = ['results-' Orig '.mat'];  
  load(fullfile(curfolder, resultsfile), 'curerr');
  errmat(:, i, 1) = curerr(end, 1:IterNum);  
  resultsfile = ['results-' Beta{i} '.mat'];  
  load(fullfile(curfolder, resultsfile), 'curerr');
  errmat(:, i, 2) = curerr(end, 1:IterNum);  
end; 

val3 = mean(errmat(:, :, 1), 1) * 100;
val4 = mean(errmat(:, :, 2), 1) * 100;

%{ ---------------------------------------------------- %}


fig = figure;
hold on;

clear LineSpec;
LineSpec{1} = 'b-x';
LineSpec{2} = 'r--x';
LineSpec{3} = 'b-o';
LineSpec{4} = 'r--o';
LineSpec{5} = 'c-s';
LineSpec{6} = 'k--d';
LineSpec{7} = 'y:^';
LineSpec{8} = 'b-.+';
LineSpec{9} = 'g-o';

xrange = 1:BetaNum;

mycolor = [0,0.447,0.741];
%errorbar(ax(1), mean(diff, 1), std(diff, 1), LineSpec{1});
h = plot(xrange, val1, LineSpec{1});
h.Color = mycolor;
h = plot(xrange, val2, LineSpec{2});
h.Color = mycolor;
%axis([0.5 8.5 0 5.2]);
%plot(xrange, betas*10, LineSpec{5});

betas = [0.2 0.15 0.1 0.05 0.03 0.02 0.02 0.02];
[ax, h1, h2] = plotyy(0, 0, xrange, betas);
legend(ax(2), 'hide');
axis(ax(1), [0.5 8.5 -0.3 5.5]);
axis(ax(2), [0.5 8.5 -0.03 0.55]);
set(ax(1),'XTickLabel', xvalues);
%set(ax(2),'YTickLabel', []);
xlabel('Dataset size, $\times 10^3$','Interpreter','latex');
ylabel('Error, \%','Interpreter','Latex');
h1.LineStyle = ':';
%mycolor = h1.Color;
h1.Color = h2.Color;
h1.Marker = 'x';
h2.LineStyle = ':';
%h2.Color = 'b';
h2.Marker = 'x';

h = plot(xrange, val3, LineSpec{3});
h.Color = mycolor;
h = plot(xrange, val4, LineSpec{4});
h.Color = mycolor;

betas = [0.01 0.005 0.003 0.003 0.002 0.002 0.002 0.002] * 10;
hold(ax(2), 'on');
h3 = plot(xrange, betas);

h3.LineStyle = ':';
%h2.Color = 'b';
h3.Marker = 'o';

%set(ax(2), 'YTick', [0 0.2]);
%ylabel(ax(2), 'Optimal $\beta$ values','Interpreter','Latex')


grid on;


legstr{1} = 'BP, no augmentation';
legstr{2} = 'IBP, no augmentation';
legstr{3} = 'Best $\beta$, no augmentation';
legstr{4} = 'BP, with augmentation';
legstr{5} = 'IBP, with augmentation';
legstr{6} = 'Best $\beta$, with augmentation';
%wlegstr{2} = 'optimal $\beta$ values';
leg = legend(legstr, 'Location', 'NorthEast');
%leg = legend([h1 h2],legstr, 'Location', s'NorthEast');

set(leg,'Interpreter','latex');
%set(fig,'Position',[400 400 500 350]);
set(fig, 'position', [700 500 440 300]);

ylabel(ax(2), 'Optimal $\beta$ values','Interpreter','Latex')