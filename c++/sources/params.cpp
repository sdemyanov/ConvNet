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

#include "params.h"

Params::Params() {
  batchsize_ = 32;
  epochs_ = 1;
  test_epochs_ = 1;
  alpha_ = 1.0;
  shift_ = 0;
  testshift_ = 0;
  beta_ = 0;
  momentum_ = 0;
  decay_ = 0;
  shuffle_ = false;
  lossfun_ = "logreg";
  normfun_ = 1;
  verbose_ = 0;
  seed_ = 0;
  fast_ = true;
  memory_ = 512; // Megabytes of additional memory for CUDNN operations
  gpu_ = 0;
}

void Params::Init(const mxArray *mx_params) {
  mexAssertMsg(mexIsStruct(mx_params), "In 'Params::Init' the array in not a struct");

  if (mexIsField(mx_params, "batchsize")) {
    batchsize_ = (size_t) mexGetScalar(mexGetField(mx_params, "batchsize"));
    mexAssertMsg(batchsize_ > 0, "Batchsize must be positive");
  }
  if (mexIsField(mx_params, "epochs")) {
    epochs_ = (size_t) mexGetScalar(mexGetField(mx_params, "epochs"));
    mexAssertMsg(epochs_ > 0, "Epochs number must be positive");
  }
  if (mexIsField(mx_params, "testepochs")) {
    test_epochs_ = (size_t) mexGetScalar(mexGetField(mx_params, "testepochs"));
    mexAssertMsg(test_epochs_ > 0, "Epochs-test number must be positive");
  }
  if (mexIsField(mx_params, "alpha")) {
    alpha_ = mexGetScalar(mexGetField(mx_params, "alpha"));
    mexAssertMsg(alpha_ >= 0, "alpha must be nonnegative");
  }
  if (mexIsField(mx_params, "shift")) {
    shift_ = mexGetScalar(mexGetField(mx_params, "shift"));
    mexAssertMsg(shift_ >= 0, "Shift must be nonnegative");
  }
  if (mexIsField(mx_params, "testshift")) {
    testshift_ = (mexGetScalar(mexGetField(mx_params, "testshift")));
    mexAssertMsg(testshift_ >= 0, "Testshift must be nonnegative");
  }
  if (mexIsField(mx_params, "beta")) {
    beta_ = mexGetScalar(mexGetField(mx_params, "beta"));
    mexAssertMsg(beta_ >= 0, "beta must be nonnegative");
    mexAssertMsg(shift_ * beta_ == 0, "Both Shift and Beta cannot be positive");
  }
  if (mexIsField(mx_params, "momentum")) {
    momentum_ = mexGetScalar(mexGetField(mx_params, "momentum"));
    mexAssertMsg(0 <= momentum_ && momentum_ < 1, "Momentum is out of range [0, 1)");
  }
  if (mexIsField(mx_params, "decay")) {
    decay_ = mexGetScalar(mexGetField(mx_params, "decay"));
    mexAssertMsg(0 <= decay_ && decay_ < 1, "Decay is out of range [0, 1)");
  }
  if (mexIsField(mx_params, "shuffle")) {
    shuffle_ = (mexGetScalar(mexGetField(mx_params, "shuffle")) > 0);
  }
  if (mexIsField(mx_params, "lossfun")) {
    lossfun_ = mexGetString(mexGetField(mx_params, "lossfun"));
    mexAssertMsg(lossfun_ == "logreg" || lossfun_ == "L-norm",
      "Unknown loss function in params");
  }
  if (mexIsField(mx_params, "normfun")) {
    normfun_ = (int) mexGetScalar(mexGetField(mx_params, "normfun"));
    mexAssertMsg(normfun_ == 1 || normfun_ == 2,
      "Normfun might be equal to 1 or 2");
  }
  if (mexIsField(mx_params, "verbose")) {
    verbose_ = (int) mexGetScalar(mexGetField(mx_params, "verbose"));
    mexAssertMsg(0 <= verbose_ && verbose_ <= 5,
      "Verbose must be from 0 to 4");
  }
  if (mexIsField(mx_params, "seed")) {
    seed_ = (int) mexGetScalar(mexGetField(mx_params, "seed"));
  }
  if (mexIsField(mx_params, "fast")) {
    fast_ = (mexGetScalar(mexGetField(mx_params, "fast")) > 0);
    // I don't know why the below is needed
    /* if (!fast_ && shift_ > 0) { // beta = 0
      beta_ = 1;
    } */
  }
  if (mexIsField(mx_params, "memory")) {
    memory_ = (size_t) mexGetScalar(mexGetField(mx_params, "memory"));
  }
  if (mexIsField(mx_params, "gpu")) {
    gpu_ = (int) mexGetScalar(mexGetField(mx_params, "gpu"));
    mexAssertMsg(0 <= gpu_ , "GPU index should be non-negative");
  }
  if (mexIsField(mx_params, "classcoefs")) {
    mexGetMatrix(mexGetField(mx_params, "classcoefs"), classcoefs_);
    mexAssertMsg(classcoefs_.size2() == 1,
      "Classcoefs should be an 1xN vector");
  }
}
