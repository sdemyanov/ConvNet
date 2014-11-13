function [] = compile(regime, varargin)

clear mex;
kMainFolder = fileparts(mfilename('fullpath'));
cd(kMainFolder);
kMainFolder = pwd;

kCPUFolder = fullfile(kMainFolder, 'c++');
kCppSource = fullfile(kCPUFolder, 'sources');
kCppInclude = fullfile(kCPUFolder, 'include');

kCudaFolder = fullfile(kCPUFolder, 'cuda');
kCudaInclude = fullfile(kCudaFolder, 'include');
kCudaSource = fullfile(kCudaFolder, 'sources');

targets = {'cnntrain_mex.cpp', ...
           'classify_mex.cpp', ...
           'genweights_mex.cpp'};
          
if (nargin >= 2)
  indices = varargin{1};
else
  indices = 1:numel(targets);
end;

cppfiles = fullfile(kCppSource, '*.cpp');
kObjFolder = fullfile(kCPUFolder, 'obj');
if (~exist(kObjFolder, 'dir'))
  mkdir(kObjFolder);
end;

% generating c++ object files
if (regime == 0) % single thread CPU
  mex(['"' cppfiles '"'], '-c', ...      
      ['-I' kCppInclude], ...
      '-outdir', kObjFolder, ...
      '-largeArrayDims');
elseif (regime == 1) % multithread CPU
  mex(['"' cppfiles '"'], '-c', ...      
      ['-I' kCppInclude], ...
      '-outdir', kObjFolder, ...
      '-largeArrayDims', ...
      'COMPFLAGS="/openmp $COMPFLAGS"', ...
      'CXXFLAGS=""\$CXXFLAGS -fopenmp""', ...
      'LDFLAGS=""\$LDFLAGS -fopenmp""');
elseif (regime == 2) % GPU
  kCudaPath = getenv('CUDA_PATH');
  if (isempty(kCudaPath)) 
    assert(0, 'Install CUDA and/or setup its path in the "CUDA_PATH" variable using "setenv"');
    %setenv('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5');
  end;
  kCudaHeaders = fullfile(kCudaPath, 'include');  
  cudaInclude = ['-I' kCudaInclude];
  cudaHeaders = ['-I' kCudaHeaders]; 
  mex(['"' cppfiles '"'], '-c', ...      
      ['-I' kCppInclude], ...
      cudaInclude, cudaHeaders, ...
      '-outdir', kObjFolder, ...
      '-largeArrayDims');
end;
disp('C++ object files created');

if (ispc)
  cppfiles = fullfile(kObjFolder, '*.obj');
elseif (isunix)  
  cppfiles = fullfile(kObjFolder, '*.o');
end;

% generating mex files

kBuildFolder = fullfile(kCPUFolder, 'build');
if (~exist(kBuildFolder, 'dir'))
  mkdir(kBuildFolder);
end;
if (regime ~= 2) % CPU
  kBuildFolder = fullfile(kBuildFolder, 'cpu');  
  if (~exist(kBuildFolder, 'dir'))
    mkdir(kBuildFolder);
  end;
  if (~exist('indices', 'var'))
    indices = 1:numel(targets);
  end;
  for i = 1 : numel(indices)
    mexfile = fullfile(kCPUFolder, targets{indices(i)});
    mex(mexfile, ['"' cppfiles '"'], ...      
        ['-I' kCppInclude], ...
        '-lut', ...
        '-largeArrayDims', ...
        '-outdir', kBuildFolder);
    disp([targets{i} ' compiled']);  
  end;
  
elseif (regime == 2) % GPU

  % setup cuda settings
  kCudaPath = getenv('CUDA_PATH');
  kCudaHeaders = fullfile(kCudaPath, 'include');  
  if (ispc)
    kVSFolder = getenv('VS100COMNTOOLS');
    if (isempty(kVSFolder)) 
      assert(0 == 1, 'Install Visual Studio and/or setup the path to its "Tools" in the "VS100COMNTOOLS" variable using "setenv"');
      %setenv('VS100COMNTOOLS', 'C:\Program Files (x86)\Microsoft Visual Studio 11.0\Common7\Tools\');
    end;    
  end;
  arch = computer;
  if (strcmp(arch, 'PCWIN32'))
    kCudaLib = fullfile(kCudaPath, 'lib', 'Win32'); 
  elseif (strcmp(arch, 'PCWIN64'))
    kCudaLib = fullfile(kCudaPath, 'lib', 'x64'); 
  end;  
  copyfile(fullfile(kCudaFolder, '*.xml'), kMainFolder, 'f');  
  cudafiles = fullfile(kCudaSource, '*.cu');  
  
  kBuildFolder = fullfile(kBuildFolder, 'gpu');
  if (~exist(kBuildFolder, 'dir'))
    mkdir(kBuildFolder);
  end;  
  for i = 1 : numel(indices)
    mexfile = fullfile(kCPUFolder, targets{indices(i)});
    mex(mexfile, ['"' cudafiles '"'], ['"' cppfiles '"'], ...      
        ['-I' kCppInclude], ...
        ['-I' kCudaInclude], ...
        ['-I' kCudaHeaders], ...
        ['-L' kCudaLib], ...
        '-lut', '-lcurand', '-lcublas', ...
        '-largeArrayDims', ...
        '-outdir', kBuildFolder);
    disp([targets{indices(i)} ' compiled']);
  end;
  delete(fullfile(kMainFolder, '*.xml'));
  
end;

end