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

#ifndef _SETTINGS_H_
#define _SETTINGS_H_

// COMP_REGIME = 0 -> CPU
// COMP_REGIME = 1 -> MULTITHREAD CPU
// COMP_REGIME = 2 -> GPU
#define COMP_REGIME 2

// USE_CUDNN = -1 -> NOT APPLICABLE
// USE_CUDNN = 0 -> NO
// USE_CUDNN = 1 -> YES
#define USE_CUDNN 0

// PRECISION = 1 -> float
// PRECISION = 2 -> double
#define PRECISION 1
// GPU version is only for float values
#if COMP_REGIME == 2  
  #define PRECISION 1
#else
  #define USE_CUDNN -1
#endif

#if PRECISION == 1
  typedef float ftype;
  #define MEX_CLASS mxSINGLE_CLASS
#elif PRECISION == 2
  typedef double ftype;  
  #define MEX_CLASS mxDOUBLE_CLASS
#endif

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
