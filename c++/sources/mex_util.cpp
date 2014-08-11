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

#include "mex_util.h"

bool mexIsStruct(const mxArray *mx_array) {	
  mexAssert(mx_array != NULL && !mxIsEmpty(mx_array), "In 'mexIsStruct' the array is NULL or empty");
  return mxIsStruct(mx_array);
}

bool mexIsCell(const mxArray *mx_array) {	
  mexAssert(mx_array != NULL && !mxIsEmpty(mx_array), "In 'mexIsCell' the array is NULL or empty");
  return mxIsCell(mx_array);
}

bool mexIsField(const mxArray *mx_array, const char *fieldname) {
	mexAssert(mexIsStruct(mx_array), "In 'mexGetField' the array in not a cell array");  
	const mxArray* mx_field = mxGetField(mx_array, 0, fieldname);
	return (mx_field != NULL);  
}

bool mexIsString(const mxArray *mx_array) {
  mexAssert(mx_array != NULL && !mxIsEmpty(mx_array), "In 'mexIsString' the array is NULL or empty");
  return mxIsChar(mx_array);
}

const mxArray* mexGetCell(const mxArray *mx_array, size_t ind) {  
  mexAssert(mexIsCell(mx_array), "In 'mexGetCell' the array in not a cell array");
  size_t numel = mexGetNumel(mx_array);
  mexAssert(ind < numel, "In 'mexGetCell' index is out of array");
  return mxGetCell(mx_array, ind);  
}

const mxArray* mexGetField(const mxArray *mx_array, const char *fieldname) {	  
  mexAssert(mexIsStruct(mx_array), "In 'mexGetField' the array in not a cell array");
  const mxArray* mx_field = mxGetField(mx_array, 0, fieldname);	  
  std::string fieldname_str(fieldname); 
	mexAssert(mx_field != NULL, fieldname + std::string(" field missing!!\n"));  
	return mx_field;  
}

size_t mexGetDimensionsNum(const mxArray *mx_array) {
  mexAssert(mx_array != NULL, "mx_array in 'mexGetDimensionsNum' is NULL");
  return (size_t) mxGetNumberOfDimensions(mx_array);
}

std::vector<size_t> mexGetDimensions(const mxArray *mx_array) {  
  //mexAssert(mx_array != NULL, "mx_array in 'mexGetDimentions' is NULL");	
  size_t dimnum = mexGetDimensionsNum(mx_array);
  std::vector<size_t> dim(dimnum);
  const mwSize *pdim = mxGetDimensions(mx_array);
  for (size_t i = 0; i < dimnum; ++i) {
    dim[i] = (size_t) pdim[i];    
  }
  size_t tmp = dim[0]; dim[0] = dim[1]; dim[1] = tmp;
  // because Matlab stores data in the reverse order
  return dim;
}

size_t mexGetNumel(const mxArray *mx_array) {
  std::vector<size_t> dim = mexGetDimensions(mx_array);
  size_t numel = 1;
  for (size_t i = 0; i < dim.size(); ++i) {
    numel *= dim[i];
  }
  return numel;
}

std::string mexGetString(const mxArray *mx_array) {
  const size_t kMaxFieldLength = 100;
  char s[kMaxFieldLength];
  mexAssert(mexIsString(mx_array), "In 'mexGetSting' mx_array in not a string!");  
  mexAssert(!mxGetString(mx_array, s, kMaxFieldLength), "Error when reading string field");
  std::string str(s);
  return str;
}

ftype* mexGetPointer(const mxArray *mx_array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetPointer' is NULL");
	mexAssert(mxGetClassID(mx_array) == MEX_CLASS,
    "In 'mexGetPointer' mx_array is of the wrong type");
  return (ftype*) mxGetData(mx_array);
}

ftype mexGetScalar(const mxArray *mx_array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetScalar' is NULL");
	mexAssert(mxIsNumeric(mx_array), "In 'mexGetScalar' mx_array is not numeric");
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
  //mexAssert(mx_array != NULL, "mx_array in 'mexGetVector' is NULL");  
  size_t numel = mexGetNumel(mx_array);  
  mexAssert(mxIsNumeric(mx_array), "In 'mexGetVector' mx_array is not numeric" );
  std::vector<ftype> vect(numel);
  if (mxGetClassID(mx_array) == mxSINGLE_CLASS) {
    float *pdata = (float*) mxGetData(mx_array);
    for (size_t i = 0; i < numel; ++i) {
      vect[i] = (ftype) pdata[i];    
    }
  } else if (mxGetClassID(mx_array) == mxDOUBLE_CLASS) {
    double *pdata = (double*) mxGetData(mx_array);
    for (size_t i = 0; i < numel; ++i) {
      vect[i] = (ftype) pdata[i];    
    }
  }  
  return vect;
}

Mat mexGetMatrix(const mxArray *mx_array) {  
  //mexAssert(mx_array != NULL, "mx_array in 'mexGetMatrix' is NULL");
  std::vector<size_t> dim = mexGetDimensions(mx_array);
  mexAssert(dim.size() == 2, "In 'GetMatrix' argument must be the 2D matrix");  
  mexAssert(mxGetClassID(mx_array) == MEX_CLASS,
    "In 'mexGetMatrix' mx_array is of the wrong type");
  ftype *pdata = (ftype*) mxGetData(mx_array);
  Mat mat(dim[1], dim[0]);
  mat.attach(pdata, dim);
  return mat;
}

mxArray* mexNewMatrix(size_t size1, size_t size2) {
  mwSize ndims = 2, dims[2];
  dims[0] = size2; dims[1] = size1; // because Matlab stores data in the reverse order
  mxArray *mx_array = mxCreateNumericArray(ndims, dims, MEX_CLASS, mxREAL);	
  return mx_array;
}

mxArray* mexSetScalar(ftype scalar) {  
  mxArray *mx_scalar = mexNewMatrix(1, 1);
	ftype *pdata = (ftype*) mxGetData(mx_scalar);
	pdata[0] = scalar;
	return mx_scalar;  
}

mxArray* mexSetVector(const std::vector<ftype> &vect) {  	
  mxArray *mx_array = mexNewMatrix(vect.size(), 1);
	ftype *pdata = (ftype*) mxGetData(mx_array);
  for (size_t i = 0; i < vect.size(); ++i) {
    pdata[i] = vect[i];
  }
	return mx_array;  
}

mxArray* mexSetMatrix(const Mat &mat) {			
  mxArray *mx_array = mexNewMatrix(mat.size1(), mat.size2());  
	ftype *pdata = (ftype*) mxGetData(mx_array);  
  mat.ToVect(pdata);  
	return mx_array;  
}

void mexSetCell(mxArray* mx_array, size_t ind, mxArray* mx_value) {
  size_t numel = mexGetNumel(mx_array);  
  mexAssert(ind < numel, "In mexSetCell the index is out of range");
  mxSetCell(mx_array, ind, mx_value);
}

mxArray* mexSetCellMat(size_t size1, size_t size2) {
  return mxCreateCellMatrix(size1, size2);
}

mxArray* mexDuplicateArray(const mxArray* mx_array) {
  return mxDuplicateArray(mx_array);
}
