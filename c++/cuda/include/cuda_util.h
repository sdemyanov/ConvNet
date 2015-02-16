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

#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include "mat_gpu.h"

void cuda_validate(MatGPU &mat);
void cuda_sign(MatGPU &mat);
void cuda_sqrt(MatGPU &mat);
void cuda_log(MatGPU &mat);
void cuda_exp(MatGPU &mat);
void cuda_sigmoid(MatGPU &mat);

void cuda_assval(MatGPU &mat, float val);
void cuda_addval(MatGPU &mat, float val);
void cuda_subval(MatGPU &mat, float val);
void cuda_multval(MatGPU &mat, float val);
void cuda_divval(MatGPU &mat, float val);

void cuda_addmat(MatGPU &mat, const MatGPU &b);
void cuda_submat(MatGPU &mat, const MatGPU &b);
void cuda_multmat(MatGPU &mat, const MatGPU &b);
void cuda_divmat(MatGPU &mat, const MatGPU &b);
void cuda_sigmder(MatGPU &mat, const MatGPU &b);

void cuda_condassign(MatGPU& mat, const MatGPU& condmat, bool incase, float threshold, float val);
void cuda_condadd(MatGPU& mat, const MatGPU& condmat, bool incase, float threshold, float val);
void cuda_condmult(MatGPU& mat, const MatGPU& condmat, bool incase, float threshold, float val);

void cuda_addvect(MatGPU &mat, const MatGPU &vect, size_t dim);
void cuda_multvect(MatGPU &mat, const MatGPU &vect, size_t dim);

void cuda_sumvect(MatGPU &mat, MatGPU &vect, size_t dim);
void cuda_maxvect(MatGPU &mat, MatGPU &vect, size_t dim);

#endif