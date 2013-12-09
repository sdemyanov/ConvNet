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

#ifndef _MEX_UTIL_H_
#define _MEX_UTIL_H_

#include "mat.h"
#include "mex_print.h"
#include <vector>

bool mexIsCell(const mxArray *mx_array);

const mxArray* mexGetCell(const mxArray *mx_array, size_t ind);

bool mexIsStruct(const mxArray *mx_array);

const mxArray* mexGetField(const mxArray *mx_array, const char *fieldname);

size_t mexGetDimensionsNum(const mxArray *mx_array);

std::vector<size_t> mexGetDimensions(const mxArray *mx_array);

size_t mexGetNumel(const mxArray *mx_array);

double mexGetScalar(const mxArray *mx_array);

std::vector<double> mexGetVector(const mxArray *mx_array);

std::vector< std::vector<double> > mexGetVector2D(const mxArray *mx_array);

void mexGetMatrix(const mxArray *mx_array, Mat &array);

void mexGetMatrix3D(const mxArray *mx_array, std::vector<Mat> &array);

std::string mexGetString(const mxArray *mx_array);

bool mexIsField(const mxArray *mx_array, const char *fieldname);

mxArray* mexSetScalar(double scalar);

mxArray* mexSetVector(const std::vector<double> &vect);

mxArray* mexSetMatrix(const Mat &mat);

#endif
