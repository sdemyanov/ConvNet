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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "mex_util.h"

static clock_t _start_timer_time = 0;

void StartTimer() {
  if (print < 2) return;
  _start_timer_time = std::clock();
}

void MeasureTime(std::string msg) {
  if (print < 2) return;
  clock_t t0 = _start_timer_time;
  clock_t t = std::clock();
  double d = double(t - t0);
  mexPrintMsg(msg, d);
}

bool mexIsStruct(const mxArray *mx_array) {
  mexAssertMsg(mx_array != NULL && !mxIsEmpty(mx_array), "In 'mexIsStruct' the array is NULL or empty");
  return mxIsStruct(mx_array);
}

bool mexIsCell(const mxArray *mx_array) {
  mexAssertMsg(mx_array != NULL && !mxIsEmpty(mx_array), "In 'mexIsCell' the array is NULL or empty");
  return mxIsCell(mx_array);
}

bool mexIsField(const mxArray *mx_array, const char *fieldname) {
  mexAssertMsg(mexIsStruct(mx_array), "In 'mexIsField' the array in not a struct");
  const mxArray* mx_field = mxGetField(mx_array, 0, fieldname);
  return (mx_field != NULL);
}

bool mexIsString(const mxArray *mx_array) {
  mexAssertMsg(mx_array != NULL && !mxIsEmpty(mx_array), "In 'mexIsString' the array is NULL or empty");
  return mxIsChar(mx_array);
}

const mxArray* mexGetCell(const mxArray *mx_array, size_t ind) {
  mexAssertMsg(mexIsCell(mx_array), "In 'mexGetCell' the array in not a cell array");
  size_t numel = mexGetNumel(mx_array);
  mexAssertMsg(ind < numel, "In 'mexGetCell' index is out of array");
  return mxGetCell(mx_array, ind);
}

const mxArray* mexGetField(const mxArray *mx_array, const char *fieldname) {
  mexAssertMsg(mexIsStruct(mx_array), "In 'mexGetField' the array in not a struct");
  const mxArray* mx_field = mxGetField(mx_array, 0, fieldname);
  std::string fieldname_str(fieldname);
  mexAssertMsg(mx_field != NULL, fieldname + std::string(" field is missing!\n"));
  return mx_field;
}

size_t mexGetDimensionsNum(const mxArray *mx_array) {
  mexAssertMsg(mx_array != NULL, "mx_array in 'mexGetDimensionsNum' is NULL");
  return (size_t) mxGetNumberOfDimensions(mx_array);
}

std::vector<size_t> mexGetDimensions(const mxArray *mx_array) {
  //mexAssertMsg(mx_array != NULL, "mx_array in 'mexGetDimentions' is NULL");
  // IMPORTANT! Returning dimensions ordered from slowest to fastest!
  // This is opposite to Matlab ordering!
  size_t dimnum = mexGetDimensionsNum(mx_array);
  std::vector<size_t> dims(dimnum);
  const mwSize *pdim = mxGetDimensions(mx_array);
  for (size_t i = 0; i < dimnum; ++i) {
    dims[dimnum - i - 1] = (size_t) pdim[i];
  }
  return dims;
}

size_t mexGetNumel(const mxArray *mx_array) {
  std::vector<size_t> dims = mexGetDimensions(mx_array);
  size_t numel = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    numel *= dims[i];
  }
  return numel;
}

std::string mexGetString(const mxArray *mx_array) {
  const size_t kMaxFieldLength = 100;
  char s[kMaxFieldLength];
  mexAssertMsg(mexIsString(mx_array), "In 'mexGetSting' mx_array in not a string!");
  mexAssertMsg(!mxGetString(mx_array, s, kMaxFieldLength), "Error when reading string field");
  std::string str(s);
  return str;
}

ftype* mexGetPointer(const mxArray *mx_array) {
  mexAssertMsg(mx_array != NULL, "mx_array in 'mexGetPointer' is NULL");
  mexAssertMsg(mxGetClassID(mx_array) == MEX_CLASS,
    "In 'mexGetPointer' mx_array is of the wrong type");
  return (ftype*) mxGetData(mx_array);
}

ftype mexGetScalar(const mxArray *mx_array) {
  mexAssertMsg(mx_array != NULL, "mx_array in 'mexGetScalar' is NULL");
  mexAssertMsg(mxIsNumeric(mx_array), "In 'mexGetScalar' mx_array is not numeric");
  if (mxGetClassID(mx_array) == mxSINGLE_CLASS) {
    float *pdata = (float*) mxGetData(mx_array);
    return (ftype) pdata[0];
  } else if (mxGetClassID(mx_array) == mxDOUBLE_CLASS) {
    double *pdata = (double*) mxGetData(mx_array);
    return (ftype) pdata[0];
  }
  return 0;
}

std::vector<ftype> mexGetVector(const mxArray *mx_array) {
  //mexAssertMsg(mx_array != NULL, "mx_array in 'mexGetVector' is NULL");
  size_t numel = mexGetNumel(mx_array);
  mexAssertMsg(mxIsNumeric(mx_array), "In 'mexGetVector' mx_array is not numeric" );
  mexAssertMsg(mxGetClassID(mx_array) == mxDOUBLE_CLASS, "In mexGetVector wrong type");
  std::vector<ftype> vect(numel);
  double *pdata = (double*) mxGetData(mx_array);
  for (size_t i = 0; i < numel; ++i) {
    vect[i] = (ftype) pdata[i];
  }
  return vect;
}

void mexGetMatrix(const mxArray *mx_array, MatCPU &mat) {
  //mexAssertMsg(mx_array != NULL, "mx_array in 'mexGetMatrix' is NULL");
  std::vector<size_t> dims = mexGetDimensions(mx_array);
  mexAssertMsg(dims.size() == 2, "In 'mexGetMatrix' argument must be the 2D matrix");
  mexAssertMsg(mxGetClassID(mx_array) == MEX_CLASS,
    "In 'mexGetMatrix' mx_array is of the wrong type");
  ftype *pdata = (ftype*) mxGetData(mx_array);
  // Order is opposite to Matlab order, i.e. true
  mat.attach(pdata, dims[0], dims[1], true);
}

Dim mexGetTensor(const mxArray *mx_array, MatCPU &mat) {
  //mexAssertMsg(mx_array != NULL, "mx_array in 'mexGetMatrix' is NULL");
  // Notice that returned dims correspond to NCHW,
  // mexGetDimensions returns dimensions in reverse order, so
  // so the matrix in Matlab should be WHCN
  std::vector<size_t> mx_dims = mexGetDimensions(mx_array);
  mexAssertMsg(mx_dims.size() <= 4, "The data array must have max 4 dimensions");
  Dim dims = {1, 1, 1, 1};
  for (size_t i = 0; i < mx_dims.size(); ++i) {
    mexAssertMsg(mx_dims[i] < INT_MAX, "Tensor is too large!");
    dims[4 - mx_dims.size() + i] = (int) mx_dims[i];
  }
  mexAssertMsg(mxGetClassID(mx_array) == MEX_CLASS,
    "In 'mexGetTensor' mx_array is of the wrong type");
  ftype *pdata = (ftype*) mxGetData(mx_array);
  // Order is opposite to Matlab order, i.e. true
  mat.attach(pdata, dims[0], dims[1] * dims[2] * dims[3], true);
  return dims;
}

mxArray* mexNewArray(const std::vector<size_t> &dimvect) {
  mexAssertMsg(kInternalOrder == true, "mexNewTensor assert");
  mwSize ndims = dimvect.size(), dims[ndims];
  // opposite order!
  for (size_t i = 0; i < dimvect.size(); ++i) {
    dims[i] = dimvect[dimvect.size() - i - 1];
  }
  mxArray *mx_array = mxCreateNumericArray(ndims, dims, MEX_CLASS, mxREAL);
  return mx_array;
}

mxArray* mexNewMatrix(size_t size1, size_t size2) {
  std::vector<size_t> dimvect(2);
  dimvect[0] = size1;
  dimvect[1] = size2;
  return mexNewArray(dimvect);
}

mxArray* mexSetScalar(ftype scalar) {
  mxArray *mx_scalar = mexNewMatrix(1, 1);
  ftype *pdata = (ftype*) mxGetData(mx_scalar);
  pdata[0] = scalar;
  return mx_scalar;
}

mxArray* mexSetVector(const std::vector<ftype> &vect) {
  mxArray *mx_array = mexNewMatrix(1, vect.size());
  ftype *pdata = (ftype*) mxGetData(mx_array);
  for (size_t i = 0; i < vect.size(); ++i) {
    pdata[i] = vect[i];
  }
  return mx_array;
}

mxArray* mexSetMatrix(const MatCPU &mat) {
  mexAssertMsg(mat.order() == true, "mexSetTensor assert");
  mxArray *mx_array = mexNewMatrix(mat.size1(), mat.size2());
  ftype *pdata = (ftype*) mxGetData(mx_array);
  MatCPU mr;
  mr.attach(pdata, mat.size1(), mat.size2(), mat.order());
  mr = mat;
  return mx_array;
}

mxArray* mexSetTensor(const MatCPU &mat, const Dim& dims) {
  mexAssertMsg(mat.order() == true, "mexSetTensor assert");
  mexAssertMsg(mat.size1() * mat.size2() == dims[0] * dims[1] * dims[2] * dims[3],
    "In mexSetTensor dimensions don't correspond to matrix");
  std::vector<size_t> dimvect(4);
  for (size_t i = 0; i < 4; ++i) {
    dimvect[i] = dims[i];
  }
  mxArray *mx_array = mexNewArray(dimvect);
  ftype *pdata = (ftype*) mxGetData(mx_array);
  MatCPU mr;
  mr.attach(pdata, mat.size1(), mat.size2(), mat.order());
  mr = mat;
  return mx_array;
}

void mexSetCell(mxArray* mx_array, size_t ind, mxArray* mx_value) {
  size_t numel = mexGetNumel(mx_array);
  mexAssertMsg(ind < numel, "In mexSetCell the index is out of range");
  mxSetCell(mx_array, ind, mx_value);
}

mxArray* mexSetCellMat(size_t size1, size_t size2) {
  return mxCreateCellMatrix(size1, size2);
}

mxArray* mexDuplicateArray(const mxArray* mx_array) {
  return mxDuplicateArray(mx_array);
}
