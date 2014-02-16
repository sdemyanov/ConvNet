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

#ifndef _FTYPE_H_
#define _FTYPE_H_

//typedef float ftype;
//const ftype kEps = 1e-4;
//#define MEX_CLASS mxSINGLE_CLASS

typedef double ftype;
const ftype kEps = 1e-8;
#define MEX_CLASS mxDOUBLE_CLASS

#define USE_MULTITHREAD 1
#if USE_MULTITHREAD == 1
  #include <omp.h>
#endif

#endif