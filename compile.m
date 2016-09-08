function [] = compile()

% THIS FUNCTION HAS NOT BEEN TESTED ! ADJUST IF NEEDED !

clear mex;
kMainFolder = fileparts(mfilename('fullpath'));
cd(kMainFolder);
kMainFolder = pwd;

kCPUFolder = fullfile(kMainFolder, 'c++');
kCppSource = fullfile(kCPUFolder, 'sources');
kCppInclude = fullfile(kCPUFolder, 'include');
cppInclude = ['-I' kCppInclude];

kCudaFolder = fullfile(kCPUFolder, 'cuda');
kCudaSource = fullfile(kCudaFolder, 'sources');
kCudaInclude = fullfile(kCudaFolder, 'include');
cudaInclude = ['-I' kCudaInclude];

targets = {'train_mex.cpp', ...
           'classify_mex.cpp', ...
           'genweights_mex.cpp'};
          
if (ispc)
  kCudaPath = getenv('CUDA_PATH');
  if (isempty(kCudaPath)) 
    assert(0==1, 'Install CUDA and/or setup its path in the "CUDA_PATH" variable using "setenv"');
    %setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5');
  end;
  arch = computer;
  if (strcmp(arch, 'PCWIN32'))
    kCudaLib = fullfile(kCudaPath, 'lib', 'Win32'); 
  elseif (strcmp(arch, 'PCWIN64'))
    kCudaLib = fullfile(kCudaPath, 'lib', 'x64'); 
  end;
  objext = '.obj';
else
  kCudaPath = '/usr/local/cuda';
  kCudaLib = fullfile(kCudaPath, 'lib64');
  objext = '.o';
end;

kCudaHeaders = fullfile(kCudaPath, 'include');  
cudaHeaders = ['-I' kCudaHeaders]; 

cppfiles = fullfile(kCppSource, '*.cpp');
% compiling cpp files
kObjFolder = fullfile(kCPUFolder, 'obj');
if (~exist(kObjFolder, 'dir'))
  mkdir(kObjFolder);
end;
mex(['"' cppfiles '"'], '-c', ...      
    cppInclude, cudaInclude, cudaHeaders, ...
    '-outdir', kObjFolder, ...
    '-largeArrayDims');
cppfiles = fullfile(kObjFolder, strcat('*', objext));
disp('C++ files compiled');

%objfiles = fullfile(kObjFolder, '*.o');

cudafiles = fullfile(kCudaSource, '*.cu');
% you can try to use this if you have Parallel Computing Toolbox installed
%{
% compiling cuda files
kCudaObjFolder = fullfile(kCudaFolder, 'obj');
if (~exist(kCudaObjFolder, 'dir'))
  mkdir(kCudaObjFolder);
end;
mexcuda(['"' cudafiles '"'], '-c', ...      
    cppInclude, cudaInclude, cudaHeaders, ...
    '-outdir', kCudaObjFolder, ...
    '-largeArrayDims');
cudafiles = fullfile(kCudaObjFolder, strcat('*.cu', objext));
disp('Cuda files compiled');
%}

% setup cuda settings
if (ispc)
  kVSFolder = getenv('VS100COMNTOOLS');
  if (isempty(kVSFolder)) 
    assert(0 == 1, 'Install Visual Studio and/or setup the path to its "Tools" in the "VS100COMNTOOLS" variable using "setenv"');
    %setenv('VS100COMNTOOLS', 'C:\Program Files (x86)\Microsoft Visual Studio 11.0\Common7\Tools\');
  end;    
end;

copyfile(fullfile(kCudaFolder, '*.xml'), kMainFolder, 'f');

% generating mex files
kBuildFolder = fullfile(kCPUFolder, 'build');
if (~exist(kBuildFolder, 'dir'))
  mkdir(kBuildFolder);
end;
indices = 1:numel(targets);
for i = 1 : numel(indices)
  mexfile = fullfile(kCPUFolder, targets{indices(i)});
  mex(mexfile, ['"' cppfiles '"'], ['"' cudafiles '"'], ...      
      cppInclude, cudaInclude, cudaHeaders, ['-L' kCudaLib], ...
      '-lut', '-lcudart', '-lcurand', '-lcublas', '-lcudnn', ... 
      '-largeArrayDims', ...
      '-outdir', kBuildFolder);
  disp([targets{indices(i)} ' compiled']);
end;
delete(fullfile(kMainFolder, '*.xml'));

end
