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

#define NARGIN 3
#define IN_L pRhs[0] // layers
#define IN_W pRhs[1] // weights
#define IN_X pRhs[2] // data

#define NARGOUT 1
#define OUT_P	pLhs[0] // predictions

int print = 0;

void mexFunction(int nLhs, mxArray* pLhs[], int nRhs, const mxArray* pRhs[]) {

  mexAssert(nRhs == NARGIN && nLhs == NARGOUT, 
    "Number of input and/or output arguments is not correct!" );
  
  Net net;
  net.InitLayers(IN_L);  
  net.InitWeights(IN_W);  
  net.Classify(IN_X, OUT_P);
  
}
