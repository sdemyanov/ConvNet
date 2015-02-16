
%datasetName = 'MNIST';
datasetName = 'CIFAR';
%datasetName = 'CIFAR-100';
%datasetName = 'SVHN';

kWorkspaceFolder = fullfile('/media/sergeyd/OS/Users/sergeyd/Workspaces', datasetName);
errfile = fullfile(kWorkspaceFolder, 'final-last', 'noise0.2.mat');
load(errfile, 'errors', 'stdev');

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

kStNum = size(errors, 1);
kItersNum = size(errors, 2);
kBetaNum = size(errors, 3);

Period = 1;
MaxIter = 20;
range = 1:Period:MaxIter;

close all;
h = figure;
hold on;
for bind = 1 : kBetaNum
  plotmat = mean(errors(range, 1:kItersNum, bind), 2) * 100;
  plot(stdev(range), plotmat', LineSpec{bind});
end;
hold off;
%set(gca,'XTickLabel', stdev); 
axis([0 0.1 15 45]);
%axis([0 0.1 45 85]);
clear legstr; n = 1;
legstr{n} = 'BP ($\beta = 0$), no d/o';  n = n + 1;
legstr{n} = 'IBP ($\beta$ = 5e-4), no d/o';  n = n + 1;
legstr{n} = 'BP ($\beta = 0$), with d/o';  n = n + 1;
legstr{n} = 'IBP ($\beta$ = 5e-5), with d/o';  n = n + 1;
legend(legstr, 'Location', 'NorthWest','Interpreter','latex');

xlabel('Noise standard deviation');
ylabel('Error, %');
set(h, 'position', [700 500 380 240]);