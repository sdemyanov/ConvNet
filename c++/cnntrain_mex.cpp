/*
Copyright (C) 2013 Sergey Demyanov. 
contact: sergey@demyanov.net
http://www.demyanov.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "net.h"

#define NARGIN_MIN 4
#define NARGIN_MAX 5
#define IN_L pRhs[0] // layers
#define IN_P pRhs[1] // params
#define IN_X pRhs[2] // data
#define IN_Y pRhs[3] // labels
#define IN_W pRhs[4] // weights


#define NARGOUT 2
#define OUT_W	pLhs[0] // weights
#define OUT_E pLhs[1] // train errors on each batch

void mexFunction(int nLhs, mxArray* pLhs[], int nRhs, const mxArray* pRhs[]) {

  mexAssert(NARGIN_MIN <= nRhs && nRhs <= NARGIN_MAX, "Number of input arguments in wrong!");
  mexAssert(nLhs == NARGOUT, "Number of output arguments is wrong!" );
  
  Net net;
  net.InitLayers(IN_L);
  if (nRhs == 5) net.SetWeights(IN_W);
  else net.SetWeights(NULL);
  net.InitParams(IN_P);
  net.Train(IN_X, IN_Y);
  net.GetWeights(OUT_W);
  net.GetTrainError(OUT_E);
  
}
