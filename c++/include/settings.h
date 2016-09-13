/*
Copyright (C) 2016 Sergey Demyanov.
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
along with this program. If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef _SETTINGS_H_
#define _SETTINGS_H_

// N - number, C - channel, H - height, W - width
// in the layout code first is slowest, last is fastest
// order = false -> size1 is fastest, corresponds to CWHN (AlexNet) layout
// order = true  -> size2 is fastest, corresponds to NCHW (CuDNN) layout

// indicates the layout of containers and maps inside them, fed from Matlab
static const bool kExternalOrder = true;

// indicates how containers and maps inside them are stored in toolbox memory
static const bool kInternalOrder = true;

// they are preferred to match, otherwise a lot of reordering is required

// PRECISION = 1 -> float
// PRECISION = 2 -> double, but it has not been tested
#define PRECISION 1

#if PRECISION == 1
  typedef float ftype;
  #define MEX_CLASS mxSINGLE_CLASS
  #define CUDNN_TYPE CUDNN_DATA_FLOAT
#elif PRECISION == 2
  typedef double ftype;
  #define MEX_CLASS mxDOUBLE_CLASS
  #define CUDNN_TYPE CUDNN_DATA_DOUBLE
#endif

#define CUDNN_LAYOUT CUDNN_TENSOR_NCHW

#define PRECISION_EPS 1e-6
static const ftype kEps = (ftype) PRECISION_EPS;

static const ftype kPi = (ftype) 3.141592654;

#ifndef MIN
  #define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif
#ifndef MAX
  #define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef DIVUP
  #define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#endif
