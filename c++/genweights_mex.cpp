/*
Copyright (C) 2014 Sergey Demyanov. 
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

#define NARGIN 2
#define IN_L pRhs[0] // layers
#define IN_S pRhs[1] // randseed

#define NARGOUT 1
#define OUT_W	pLhs[0] // weights

int print = 0;

void mexFunction(int nLhs, mxArray* pLhs[], int nRhs, const mxArray* pRhs[]) {

  mexAssert(nRhs == NARGIN, "Number of input arguments in wrong!");
  mexAssert(nLhs == NARGOUT, "Number of output arguments is wrong!" );
  
  size_t seed = (size_t) mexGetScalar(IN_S);
  std::srand(seed);
  
  Net net;
  net.InitLayers(IN_L);  
  net.InitWeights(NULL, OUT_W);
  
}
