function createvcvars
    try
        SDKDIR = winqueryreg('HKEY_LOCAL_MACHINE','SOFTWARE\Microsoft\Microsoft SDKs\Windows\v7.1','InstallationFolder');
    catch %#ok
        error('Windows SDK 7.1 Registry Key not found!')
    end
    SETENVBAT = fullfile(SDKDIR,'Bin','SetEnv.cmd');
    if (exist(SETENVBAT,'file')==0)
        error('SetEnv script of Windows SDK 7.1 not found');
    end
    VSDIR = getenv('VS100COMNTOOLS');
    if (isempty(VSDIR))
        error('Microsoft Visual C++ 2010 Tools Directory not found')
    end
    VCDIR = fullfile(VSDIR,'..','..','VC');
    CLEXE = fullfile(VCDIR,'bin','amd64','cl.exe');
    VCVARS64 = fullfile(VCDIR,'bin','amd64','vcvars64.bat');
    if (exist(CLEXE,'file')==0)
        error('Microsoft Visual C++ 2010 64-bit Compiler not found')
    end
    fprintf('Found SDK in:\t\t\t%s\nFound C++ Compiler in:\t%s\nVCVARS64.BAT to create:\t%s\n',SDKDIR,CLEXE,VCVARS64);
    r = input('Do you want to create vcvars64.bat [Y/n]? ','s');
    if (strcmpi(r,'n'))
       disp('Aborting')
       return
    end
    if (exist(VCVARS64,'file')~=0)
        r = input('vcvars64.bat already exists do you want to overwrite [y/N]? ','s');
        if (~strcmpi(r,'y'))
           disp('Aborting')
           return
        end
    end
    fprintf('Creating %s\n',VCVARS64);
    f = fopen(VCVARS64,'wt');
    if (f<0)
        error('Error creating %s',VCVARS64);
    end
    fprintf('Writing "CALL %s /x64"\n',SETENVBAT);
    fprintf(f,'CALL %s /x64\n',SETENVBAT);
    fclose(f);
    fprintf('Done!\n');
    