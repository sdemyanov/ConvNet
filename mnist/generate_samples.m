kWorkspaceFolder = 'C:\Users\sergeyd\Workspaces\Invariance\MNIST';

is_train = 1;
if (is_train == 1)
  load(fullfile(kWorkspaceFolder, 'mnist.mat'), 'TrainX', 'TrainY');
  CurX = TrainX;
else
  load(fullfile(kWorkspaceFolder, 'mnist.mat'), 'TestX', 'TestY');
  load(fullfile(kWorkspaceFolder, 'mnist_ext.mat'), 'TrainX', 'TrainY');
  CurX = TestX;
end;

kXsize = [size(CurX, 1) size(CurX, 2)];
kCurNum = size(CurX, 3);
kNewSize = [24 24];
CurXnew = zeros([kNewSize 1*kCurNum]); 

kShift = [2 2];
for i = 1 : kCurNum
  %imshow(TrainX(:,:,i));
  if (is_train == 1) 
    CurXnew(:,:,(i-1)*5 + 1) = CurX((kXsize(1)-kNewSize(1))/2+1:(kXsize(1)+kNewSize(1))/2,...
                                        (kXsize(2)-kNewSize(2))/2+1:(kXsize(2)+kNewSize(2))/2, i);  
    %imshow(TrainXnew(:,:,(i-1)*5 + 1));
    CurXnew(:,:,(i-1)*5 + 2) = CurX((kXsize(1)-kNewSize(1))/2+1-kShift(1):(kXsize(1)+kNewSize(1))/2-kShift(1),...
                                        (kXsize(2)-kNewSize(2))/2+1-kShift(2):(kXsize(2)+kNewSize(2))/2-kShift(2), i);
    %imshow(TrainXnew(:,:,(i-1)*5 + 2));
    CurXnew(:,:,(i-1)*5 + 3) = CurX((kXsize(1)-kNewSize(1))/2+1-kShift(1):(kXsize(1)+kNewSize(1))/2-kShift(1),...
                                        (kXsize(2)-kNewSize(2))/2+1+kShift(2):(kXsize(2)+kNewSize(2))/2+kShift(2), i);
    %imshow(TrainXnew(:,:,(i-1)*5 + 3));
    CurXnew(:,:,(i-1)*5 + 4) = CurX((kXsize(1)-kNewSize(1))/2+1+kShift(1):(kXsize(1)+kNewSize(1))/2+kShift(1),...
                                        (kXsize(2)-kNewSize(2))/2+1-kShift(2):(kXsize(2)+kNewSize(2))/2-kShift(2), i);
    %imshow(TrainXnew(:,:,(i-1)*5 + 4));
    CurXnew(:,:,(i-1)*5 + 5) = CurX((kXsize(1)-kNewSize(1))/2+1+kShift(1):(kXsize(1)+kNewSize(1))/2+kShift(1),...
                                        (kXsize(2)-kNewSize(2))/2+1+kShift(2):(kXsize(2)+kNewSize(2))/2+kShift(2), i);  
    %imshow(TrainXnew(:,:,(i-1)*5 + 5));
  else
    CurXnew(:,:,i) = CurX((kXsize(1)-kNewSize(1))/2+1:(kXsize(1)+kNewSize(1))/2,...
                                    (kXsize(2)-kNewSize(2))/2+1:(kXsize(2)+kNewSize(2))/2, i);
  end;
end;

if (is_train == 1)
  TrainX = CurXnew;
  addpath(fullfile('..', 'matlab'));
  TrainY = expand(TrainY, [5 1]);
  %save(fullfile(kWorkspaceFolder, 'mnist_ext.mat'), 'TrainX', 'TrainY');
else
  TestX = CurXnew;
  %save(fullfile(kWorkspaceFolder, 'mnist_ext.mat'), 'TrainX', 'TrainY', 'TestX', 'TestY');
end;
