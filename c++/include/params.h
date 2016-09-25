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

#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "mex_util.h"

class Params {

public:
  size_t batchsize_;
  size_t epochs_;
  size_t test_epochs_;
  ftype alpha_;
  ftype shift_;
  ftype testshift_;
  ftype beta_;
  ftype decay_;
  ftype momentum_;
  bool shuffle_;
  std::string lossfun_;
  int normfun_;
  int verbose_;
  int seed_;
  bool fast_;
  size_t memory_;
  int gpu_;
  MatCPU classcoefs_;

  Params();

  void Init(const mxArray *params);

};

#endif
