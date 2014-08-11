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

#ifndef _MEX_UTIL_H_
#define _MEX_UTIL_H_

#include "mat.h"

#if defined (_WIN32)
  #define NOMINMAX
  #include <windows.h>
#elif defined (__linux__)
  #include <unistd.h>
#endif

#ifdef __cplusplus 
  extern "C" bool utIsInterruptPending();
#else
  extern bool utIsInterruptPending();
#endif

bool mexIsStruct(const mxArray *mx_array);
bool mexIsCell(const mxArray *mx_array);
bool mexIsField(const mxArray *mx_array, const char *fieldname);
bool mexIsString(const mxArray *mx_array);

const mxArray* mexGetCell(const mxArray *mx_array, size_t ind);
const mxArray* mexGetField(const mxArray *mx_array, const char *fieldname);

size_t mexGetDimensionsNum(const mxArray *mx_array);
std::vector<size_t> mexGetDimensions(const mxArray *mx_array);
size_t mexGetNumel(const mxArray *mx_array);

std::string mexGetString(const mxArray *mx_array);

ftype* mexGetPointer(const mxArray *mx_array);
ftype mexGetScalar(const mxArray *mx_array);
std::vector<ftype> mexGetVector(const mxArray *mx_array);
Mat mexGetMatrix(const mxArray *mx_array);

mxArray* mexNewMatrix(size_t size1, size_t size2);
mxArray* mexSetScalar(ftype scalar);
mxArray* mexSetVector(const std::vector<ftype> &vect);
mxArray* mexSetMatrix(const Mat &mat);
mxArray* mexSetCellMat(size_t size1, size_t size2);
void mexSetCell(mxArray* mx_array, size_t ind, mxArray* mx_value);

mxArray* mexDuplicateArray(const mxArray* mx_array);

#endif
