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

#ifndef _MAT_H_
#define _MAT_H_

#include "mex_print.h"
#include <cmath>
#include <ctime>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

class Mat{

public:
  Mat() {};
  Mat(const std::vector<size_t> &newsize);
  Mat(size_t size1, size_t size2);  
  Mat& resize(const std::vector<size_t> &newsize);  
  Mat& resize(size_t size1, size_t size2);  
  Mat& init(const std::vector<size_t> &newsize, double val);  
  Mat& init(size_t size1, size_t size2, double val);  
  Mat& assign(double val);
  Mat& AddVect(const Mat &vect, size_t dim);
  Mat& MultVect(const std::vector<double> &vect, size_t dim); // ElementWise!
  Mat& FromVect(const std::vector<double> &vect, const std::vector<size_t> &newsize);
  std::vector<double> ToVect() const;
  Mat& Rand();
  double& operator () (size_t ind);
  const double& operator () (size_t ind) const;
  double& operator () (size_t ind1, size_t ind2);
  const double& operator () (size_t ind1, size_t ind2) const;
  Mat& operator = (const Mat &mat);  
  Mat& operator += (const Mat &a);
  Mat& operator -= (const Mat &a);
  Mat& operator += (double a);
  Mat& operator -= (double a);
  Mat& operator *= (double a);
  Mat& operator /= (double a);    
  size_t size1() const;
  size_t size2() const;
  double Sum() const;  
  void Sum(size_t dim, Mat& vect) const;  
  void Mean(size_t dim, Mat& vect) const;
  std::vector<size_t> MaxInd(size_t dim) const;
  void Filter(const Mat& kernel, Mat &filtered, bool type) const;
  void SubMat(const std::vector<size_t> ind, size_t dim, Mat &submat) const;
  Mat& Sigmoid();
  Mat& SigmDer(const Mat& val);  
  void MeanScale(const std::vector<size_t> &scale, Mat &scaled) const;
  void MeanScaleDer(const std::vector<size_t> &scale, Mat &scaled) const ;  
  void MaxScale(const std::vector<size_t> &scale, Mat &scaled) const;  
  void MaxScaleDer(const std::vector<size_t> &scale, const Mat &val, const Mat &prev_val, Mat &scaled) const;
  Mat& ReshapeFrom(const std::vector< std::vector<Mat> > &squeezed);  
  void ReshapeTo(std::vector< std::vector<Mat> > &squeezed, size_t outputmaps, 
                 size_t batchsize, const std::vector<size_t> &mapsize) const;  
  void MaxTrim(Mat &trimmed, std::vector<size_t> &coords) const;
  void MaxRestore(Mat &restored, const std::vector<size_t> &coords) const;
  
  Mat& Trans();
  Mat& Sign();
  Mat& ElemProd(const Mat &a);
  Mat& ElemMax(double a);
  Mat& CondAssign(const Mat &condmat, double threshold, bool incase, double a);
  Mat& CondAdd(const Mat &condmat, double threshold, bool incase, double a);
  Mat& CondProd(const Mat &condmat, double threshold, bool incase, double a);
  friend void Trans(const Mat &a, Mat &c);
  friend void Prod(const Mat &a, const Mat &b, Mat &c);
  friend void Sum(const Mat &a, const Mat &b, Mat &c);
  
private:
  boost::numeric::ublas::matrix<double> mat_;
  
};

#endif