
#include <mex.h>
#include <algorithm>

#define NARGIN 2
#define IN_I pRhs[0] // image
#define IN_S pRhs[1] // scale

#define NARGOUT 1
#define OUT_F	pLhs[0] // filtered

inline void mexAssert(bool b, char *msg) {
  if (!b) {		
		mexErrMsgTxt(msg);
	}
}

#include <string>
inline void mexPrintMsg(std::string msg, long double x) {
  mexPrintf((msg + ": " + std::to_string(x) + "\n").c_str());
  mexEvalString("drawnow;");
}

void mexFunction(int nLhs, mxArray* pLhs[], int nRhs, const mxArray* pRhs[]) {
  
  mexAssert(nRhs == NARGIN, "Number of input arguments in wrong!");
  mexAssert(nLhs == NARGOUT, "Number of output arguments is wrong!" );

  mexAssert(IN_I != NULL, "Image array is NULL");  
  mexAssert(mxIsNumeric(IN_I), "Image is not numeric!" );  
  int imdimnum = (int) mxGetNumberOfDimensions(IN_I);
  mexAssert(imdimnum == 2 || imdimnum == 3, "The image must have 2 or 3 dimensions");
  int imdim[2];
  const mwSize *pdim = mxGetDimensions(IN_I);  
  imdim[0] = (int) pdim[0];
  imdim[1] = (int) pdim[1];
  int imnum = 1;
  if (imdimnum == 3) {
    imnum = pdim[2];
  }    
  double *pim = mxGetPr(IN_I);
  
  mexAssert(IN_S != NULL, "Image array is NULL");  
  mexAssert(mxIsNumeric(IN_S), "Scale is not numeric!" );
  int scdimnum = (int) mxGetNumberOfDimensions(IN_S);
  mexAssert(scdimnum == 2, "The scale parameter has more than 2 dimensions");
  pdim = mxGetDimensions(IN_S);
  mexAssert((int) pdim[0] == 1 || (int) pdim[1] == 1, "The scale parameter must be a vector");
  double *psc = mxGetPr(IN_S);
  
  int scale[2];
  scale[0] = (int) psc[0];
  scale[1] = (int) psc[1];
  
  mwSize scdim[3];
  scdim[0] = ceil((double) imdim[0] / scale[0]);
  scdim[1] = ceil((double) imdim[1] / scale[1]);
  scdim[2] = imnum;    
  const mwSize *mwdimc = scdim;
  
  mxArray *mx_filtered = mxCreateNumericArray(imdimnum, mwdimc, mxDOUBLE_CLASS, mxREAL);
  double *pfilt = mxGetPr(mx_filtered);
  for (int k = 0; k < scdim[2]; ++k) {
    for (int i = 0; i < scdim[0]; ++i) {
      for (int j = 0; j < scdim[1]; ++j) {
        int maxu = std::min(scale[0], (int) imdim[0] - i*scale[0]);
        int maxv = std::min(scale[1], (int) imdim[1] - j*scale[1]);
        int filtind = k*scdim[0]*scdim[1] + j*scdim[0] + i;
        int imind = k*imdim[0]*imdim[1] + j*scale[1]*imdim[0] + i*scale[0];
        pfilt[filtind] = pim[imind];        
        for (int u = 0; u < maxu; ++u) {
          for (int v = 0; v < maxv; ++v) {            
            if (pfilt[filtind] < pim[imind + v*imdim[0] + u]) {
              pfilt[filtind] = pim[imind + v*imdim[0] + u];            
            }            
          }
        }
      }
    }
  }
  OUT_F = mx_filtered;
}
