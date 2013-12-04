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

#include "mex_util.h"

const mxArray* mexGetField(const mxArray *mx_array, const char *fieldname) {	
  std::string fieldname_str(fieldname);
  std::string errmsg = "mx_array is NULL or empty, " + fieldname_str + " field\n";  
  mexAssert(mx_array != NULL && !mxIsEmpty(mx_array), errmsg);
  const mxArray* mx_field = mxGetField(mx_array, 0, fieldname);	  
  errmsg = fieldname_str + " field missing!!\n";   
	mexAssert(mx_field != NULL, errmsg);  
	return mx_field;  
}

size_t mexGetDimensionsNum(const mxArray *mx_array) {
  mexAssert(mx_array != NULL, "mx_array in 'mexGetDimensionsNum' is NULL");
  return (size_t) mxGetNumberOfDimensions(mx_array);
}

std::vector<size_t> mexGetDimensions(const mxArray *mx_array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetDimentions' is NULL");	
  size_t dimnum = mexGetDimensionsNum(mx_array);
  std::vector<size_t> dim(dimnum);
  const mwSize *pdim = mxGetDimensions(mx_array);
  for (size_t i = 0; i < dimnum; ++i) {
    dim[i] = (size_t) pdim[i];    
  }
  return dim;
}

double mexGetScalar(const mxArray *mx_array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetScalar' is NULL");
	mexAssert(mxIsNumeric(mx_array), "mx_array is not numeric" );  
  return mxGetScalar(mx_array);
}  

std::vector<double> mexGetVector(const mxArray *mx_array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetVector' is NULL");
  std::vector<size_t> dim = mexGetDimensions(mx_array);
  size_t numel = 1;
  for (size_t i = 0; i < dim.size(); ++i) {
    numel *= dim[i];
  }   
  std::vector<double> vect(numel);
  double *pdata = mxGetPr(mx_array);
  for (size_t i = 0; i < numel; ++i) {
    vect[i] = pdata[i];    
  }
  return vect;
}

template <typename T>
std::vector< std::vector<T> > mexGetVector2D(const mxArray *mx_array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetVector2D' is NULL");
  std::vector<size_t> dim = mexGetDimensions(mx_array);
  mexAssert(dim.size() == 2, "GetVector2D' argument must be the 2D array");
  std::vector< std::vector<T> > mat(dim[0]);
  double *pdata = mxGetPr(mx_array);
  for (size_t i = 0; i < dim[0]; ++i) {
    mat[i].assign(dim[1], 0);
    for (size_t j = 0; j < dim[1]; ++j) {
      mat[i][j] = (T) pdata[j*dim[0] + i];
    }
  }
  return mat;  
}

void mexGetMatrix(const mxArray *mx_array, Mat &array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetMatrix' is NULL");
  std::vector<size_t> dim = mexGetDimensions(mx_array);
  mexAssert(dim.size() == 2, "GetMatrix' argument must be the 2D array");
  array.resize(dim);  
  double *pdata = mxGetPr(mx_array);
  for (size_t i = 0; i < dim[0]; ++i) {
    for (size_t j = 0; j < dim[1]; ++j) {
      array(i, j) = (double) pdata[j*dim[0] + i];
    }
  }   
}

void mexGetMatrix3D(const mxArray *mx_array, std::vector<Mat> &array) {  
  mexAssert(mx_array != NULL, "mx_array in 'mexGetMatrix3D' is NULL");
  std::vector<size_t> dim = mexGetDimensions(mx_array);
  mexAssert(dim.size() == 3, "GetMatrix3D' argument must be the 3D array");
  array.resize(dim[2]);  
  for (size_t k = 0; k < dim[2]; ++k) {
    array[k].resize(dim);    
    double *pdata = mxGetPr(mx_array);
    for (size_t i = 0; i < dim[0]; ++i) {
      for (size_t j = 0; j < dim[1]; ++j) {
        array[k](i, j) = (double) pdata[k*dim[0]*dim[1] + j*dim[0] + i];
      }
    }
  }    
}

std::string mexGetString(const mxArray *mx_array) {
  mexAssert(mx_array != NULL, "mx_array in 'mexGetString' is NULL");
  const size_t kMaxFieldLength = 100;
  char s[kMaxFieldLength];
  mexAssert(mxIsChar(mx_array), "In 'mexGetSting' mx_array in not a string!");  
  mexAssert(!mxGetString(mx_array, s, kMaxFieldLength), "Error when reading string field\n");
  std::string str(s);
  return str;
}

bool mexIsField(const mxArray *mx_array, const char *fieldname) {	
  std::string fieldname_str(fieldname);
  std::string errmsg = "mx_array is NULL or empty, " + fieldname_str + " field\n";  
  mexAssert(mx_array != NULL && !mxIsEmpty(mx_array), errmsg);
	const mxArray* mx_field = mxGetField(mx_array, 0, fieldname);
	return (mx_field != NULL);  
}

mxArray* mexSetScalar(double scalar) {  
	mxArray *mx_scalar = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *pdata = mxGetPr(mx_scalar);
	pdata[0] = scalar;
	return mx_scalar;  
}

mxArray* mexSetVector(const std::vector<double> &vect) {  
	mxArray *mx_array = mxCreateDoubleMatrix(vect.size(), 1, mxREAL);
	double *pdata = mxGetPr(mx_array);
  for (size_t i = 0; i < vect.size(); ++i) {
    pdata[i] = (double) vect[i];
  }
	return mx_array;  
}

mxArray* mexSetMatrix(const Mat &mat) {  
	mxArray *mx_array = mxCreateDoubleMatrix(mat.size1(), mat.size2(), mxREAL);
	double *pdata = mxGetPr(mx_array);
  for (size_t i = 0; i < mat.size1(); ++i) {
    for (size_t j = 0; j < mat.size2(); ++j) {
      pdata[j*mat.size1() + i] = mat(i, j);
    }
  }
	return mx_array;  
}
