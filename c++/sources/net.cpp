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

#include "net.h"

Net::Net(const mxArray *mx_params) {
  mexAssertMsg(PRECISION == 1, "In the GPU version PRECISION should be 1");
  if (params_.verbose_ >= 1) {
    mexPrintMsg("Start params initialization...");
  }
  params_.Init(mx_params);
  MatGPU::InitCuda(params_.gpu_);
  MatGPU::SetMemoryLimit(params_.memory_);
  MatGPU::InitRand(params_.seed_);
  std::srand((unsigned int) params_.seed_);
  if (!params_.classcoefs_.empty()) {
    classcoefs_.resize(params_.classcoefs_.size1(), params_.classcoefs_.size2());
    classcoefs_ = params_.classcoefs_;
  }
  if (params_.verbose_ >= 1) {
    mexPrintMsg("Params initialization finished");
  }
}

void Net::InitLayers(const mxArray *mx_layers) {
  if (params_.verbose_ >= 1) {
    mexPrintMsg("Start layers initialization...");
  }
  size_t layers_num = mexGetNumel(mx_layers);
  layers_.resize(layers_num);
  first_trained_ = layers_num;
  Layer *prev_layer = NULL;
  for (size_t i = 0; i < layers_num; ++i) {
    const mxArray *mx_layer = mexGetCell(mx_layers, i);
    std::string layer_type = mexGetString(mexGetField(mx_layer, "type"));
    if (i == 0) {
      mexAssertMsg(layer_type == "input", "The first layer must be the 'input' type");
    }
    if (layer_type == "input") {
      layers_[i] = new LayerInput();
    } else if (layer_type == "full") {
      layers_[i] = new LayerFull();
    } else if (layer_type == "jitt") {
      layers_[i] = new LayerJitt();
    } else if (layer_type == "conv") {
      layers_[i] = new LayerConv();
    } else if (layer_type == "deconv") {
      layers_[i] = new LayerDeconv();
    } else if (layer_type == "pool") {
      layers_[i] = new LayerPool();
    } else {
      mexAssertMsg(false, layer_type + " - unknown layer type");
    }
    if (params_.verbose_ >= 1) {
     mexPrintMsg("\nInitializing layer of type", layer_type);
    }
    layers_[i]->InitGeneral(mx_layer);
    layers_[i]->Init(mx_layer, prev_layer);
    mexAssertMsg(layers_[i]->function_ != "soft" || i == layers_num - 1,
                "Softmax function may be only on the last layer");
    if (layers_[i]->NumWeights() > 0 && layers_[i]->lr_coef_ > 0) {
      if (first_trained_ > i) {
        first_trained_ = i;
      }
    }
    prev_layer = layers_[i];
    if (params_.verbose_ >= 1) {
      if (layers_[i]->NumWeights() > 0) {
        mexPrintMsg("Kernels:");
        mexPrintInt("Output channels", layers_[i]->filters_.dims(0));
        mexPrintInt("Input channels", layers_[i]->filters_.dims(1));
        mexPrintInt("Height", layers_[i]->filters_.dims(2));
        mexPrintInt("Width", layers_[i]->filters_.dims(3));
      }
      if (layers_[i]->add_bias_) {
        mexPrintMsg("Bias added");
      } else {
        mexPrintMsg("Bias not added");
      }
      if (layers_[i]->NumWeights() > 0 && layers_[i]->lr_coef_ > 0) {
        mexPrintMsg("Trainable");
      } else {
        mexPrintMsg("Fixed");
      }
      mexPrintMsg("Mapsize:");
      mexPrintInt("Channels", layers_[i]->dims_[1]);
      mexPrintInt("Height", layers_[i]->dims_[2]);
      mexPrintInt("Width", layers_[i]->dims_[3]);
    }
  }
  if (params_.verbose_ >= 1) {
    mexPrintInt("First trained layer", first_trained_);
  }
  if (params_.lossfun_ == "logreg") {
    Layer *lastlayer = layers_.back();
    if (lastlayer->function_ != "soft") {
      mexPrintMsg("WARNING: logreg loss is used with non-softmax last layer");
    }
  }
  //mexAssertMsg(layer_type == "full", "The last layer must be the type of 'f'");
  //mexAssertMsg(layers_.back()->function_ == "soft" || layers_.back()->function_ == "sigm",
  //          "The last layer function must be either 'soft' or 'sigm'");
  if (params_.verbose_ >= 1) {
    mexPrintMsg("Layers initialization finished");
  }
}

void Net::Classify(const mxArray *mx_data, const mxArray *mx_labels, mxArray *&mx_pred) {

  if (params_.verbose_ >= 3) {
    mexPrintMsg("Start classification...");
  }
  ReadData(mx_data);
  ReadLabels(mx_labels);
  size_t test_num = data_.size1();
  preds_.resize(test_num, layers_.back()->length());
  preds_.set_order(true);
  MatCPU curpreds;
  if (params_.test_epochs_ > 1) {
    curpreds.resize(test_num, layers_.back()->length());
    curpreds.set_order(true);
    preds_.assign(0);
  } else {
    curpreds.attach(preds_);
  }
  size_t numbatches = DIVUP(test_num, params_.batchsize_);
  MatGPU data_batch, labels_batch, pred_batch, coef_batch;
  for (size_t epoch = 0; epoch < params_.test_epochs_; ++epoch) {
    size_t offset = 0;
    for (size_t batch = 0; batch < numbatches; ++batch) {
      size_t batchsize = std::min(test_num - offset, params_.batchsize_);
      data_batch.resize(batchsize, data_.size2());
      SubSet(data_, data_batch, offset, true);
      InitActiv(data_batch);
      Forward(pred_batch, PassNum::ForwardTest, GradInd::Nowhere);
      if (params_.testshift_ > 0) {
        ftype loss1;
        labels_batch.resize(batchsize, labels_.size2());
        SubSet(labels_, labels_batch, offset, true);
        if (!classcoefs_.empty()) {
          coef_batch.resize(batchsize, 1);
          Prod(labels_batch, false, classcoefs_, false, coef_batch);
        }
        InitDeriv(labels_batch, coef_batch, loss1);
        // shift and beta cannot be positive together
        Backward(PassNum::Backward, GradInd::Nowhere);
        InitActivAT(params_.testshift_, 1); // L1-norm adversarial loss
        Forward(pred_batch, PassNum::ForwardTest, GradInd::Nowhere);
      }
      SubSet(curpreds, pred_batch, offset, false);
      offset += batchsize;
      if (params_.verbose_ >= 4) {
        mexPrintInt("Test epoch", epoch + 1);
        mexPrintInt("Test batch", batch + 1);
      }
    }
    if (params_.test_epochs_ > 1) {
      preds_ += curpreds;
    }
    if (params_.verbose_ >= 3) {
      mexPrintInt("Test epoch", epoch + 1);
    }
  }
  if (params_.test_epochs_ > 1) {
    preds_ /= (ftype) params_.test_epochs_;
  }
  //preds_.ReorderMaps(kInternalOrder, kExternalOrder);
  preds_.Reorder(kExternalOrder);
  Dim pred_dims = layers_.back()->dims_;
  pred_dims[0] = test_num;
  mx_pred = mexSetTensor(preds_, pred_dims);
  if (params_.verbose_ >= 3) {
    mexPrintMsg("Classification finished");
  }
}

void Net::Train(const mxArray *mx_data, const mxArray *mx_labels) {

  if (params_.verbose_ >= 3) {
    mexPrintMsg("Start training...");
  }
  ReadData(mx_data);
  ReadLabels(mx_labels);

  size_t train_num = data_.size1();
  size_t numbatches = DIVUP(train_num, params_.batchsize_);
  losses_.resize(2, params_.epochs_);
  losses_.assign(0);
  MatGPU data_batch, labels_batch, pred_batch, coef_batch, empty_batch;
  for (size_t epoch = 0; epoch < params_.epochs_; ++epoch) {
    //print = 1;
    if (params_.shuffle_) {
      Shuffle(data_, labels_);
    }
    StartTimer();
    //MatGPU::StartCudaTimer();
    size_t offset = 0;
    for (size_t batch = 0; batch < numbatches; ++batch) {
      size_t batchsize = std::min(train_num - offset, params_.batchsize_);
      data_batch.resize(batchsize, data_.size2());
      labels_batch.resize(batchsize, labels_.size2());
      SubSet(data_, data_batch, offset, true);
      SubSet(labels_, labels_batch, offset, true);
      if (!classcoefs_.empty()) {
        coef_batch.resize(batchsize, 1);
        Prod(labels_batch, false, classcoefs_, false, coef_batch);
      }
      ftype loss1 = 0, loss2 = 0;
      InitActiv(data_batch);
      Forward(pred_batch, PassNum::Forward, GradInd::Nowhere);
      InitDeriv(labels_batch, coef_batch, loss1);
      losses_(0, epoch) += loss1;
      // shift and beta cannot be positive together
      if (params_.shift_ > 0) {
        Backward(PassNum::Backward, GradInd::Nowhere);
        InitActivAT(params_.shift_, params_.normfun_);
        Forward(pred_batch, PassNum::Forward, GradInd::Nowhere);
        InitDeriv(labels_batch, coef_batch, loss2);
      }
      Backward(PassNum::Backward, GradInd::First);
      if (params_.shift_ == 0 && params_.beta_ > 0) {
        InitActivIBP(loss2, 2);
        if (params_.fast_) {
          Forward(pred_batch, PassNum::ForwardLinear, GradInd::Second);
        } else {
          Forward(pred_batch, PassNum::ForwardLinear, GradInd::Nowhere);
          labels_batch.assign(0);
          std::string lf = params_.lossfun_;
          params_.lossfun_ = "L-norm";
          // we don't multiply on the coef_batch again here
          // as the gradients are already multiplied on the first pass
          InitDeriv(labels_batch, empty_batch, loss2);
          Backward(PassNum::BackwardLinear, GradInd::Second);
          params_.lossfun_ = lf;
        }
      }
      losses_(1, epoch) += loss2;
      UpdateWeights();
      offset += batchsize;
      if (params_.verbose_ >= 4) {
        mexPrintInt("Epoch", epoch + 1);
        mexPrintInt("Batch", batch + 1);
      }
    } // batch
    //MatGPU::MeasureCudaTime("totaltime");
    MeasureTime("totaltime");
    if (params_.verbose_ >= 3) {
      mexPrintInt("Epoch", epoch + 1);
    }
  } // epoch
  losses_ /= (ftype) numbatches;
  if (params_.verbose_ >= 3) {
    mexPrintMsg("Training finished");
  }
}

void Net::Forward(MatGPU &pred, PassNum passnum, GradInd gradind) {
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Forward pass started");
    int gi = 0;
    if (gradind == GradInd::First) {
      gi = 1;
    } else if (gradind == GradInd::Second) {
      gi = 2;
    }
    mexPrintInt("Computing gradients", gi);
  }
  size_t batchsize = layers_[0]->activ_mat_.size1();
  Layer *prev_layer = NULL;
  for (size_t i = first_layer_; i < layers_.size(); ++i) {
    /*
    if (layers_[i]->type_ == "jitt") {
      first_layer_ = i;
    } */
    if (params_.verbose_ >= 4) {
      mexPrintMsg("Forward pass for layer", layers_[i]->type_);
    }
    if (gradind != GradInd::Nowhere && layers_[i]->lr_coef_ > 0) {
      layers_[i]->WeightGrads(prev_layer, gradind);
      // no BiasGrads on the forward pass
    }
    layers_[i]->ResizeActivMat(batchsize, passnum);
    layers_[i]->TransformForward(prev_layer, passnum);
    layers_[i]->AddBias(passnum);
    layers_[i]->Nonlinear(passnum);
    layers_[i]->DropoutForward(passnum);
    if (params_.verbose_ >= 5) {
      mexPrintMsg("ActivSum", layers_[i]->activ_mat_.sum());
    }
    prev_layer = layers_[i];
    if (utIsInterruptPending()) {
      mexAssert(false);
    }
  }
  pred.attach(layers_.back()->activ_mat_);
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Forward pass finished");
  }
}

void Net::Backward(PassNum passnum, GradInd gradind) {
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Backward pass started");
    int gi = 0;
    if (gradind == GradInd::First) {
      gi = 1;
    } else if (gradind == GradInd::Second) {
      gi = 2;
    }
    mexPrintInt("Computing gradients", gi);
  }
  for (size_t j = first_layer_; j < layers_.size(); ++j) {
    size_t i = first_layer_ + layers_.size() - 1 - j;
    if (params_.verbose_ >= 4) {
      mexPrintMsg("Backward pass for layer", layers_[i]->type_);
    }
    if (params_.verbose_ >= 5) {
      mexPrintMsg("DerivSum", layers_[i]->deriv_mat_.sum());
    }
    layers_[i]->DropoutBackward();
    if (layers_[i]->function_ != "soft" || params_.lossfun_ != "logreg") {
      // special case, final derivaties are already computed in InitDeriv
      layers_[i]->Nonlinear(passnum);
    }
    if (gradind != GradInd::Nowhere && layers_[i]->lr_coef_ > 0) {
      layers_[i]->BiasGrads(passnum, gradind);
      if (i > 0) {
        layers_[i]->WeightGrads(layers_[i-1], gradind);
      }
    }
    if (params_.beta_ == 0 && params_.shift_ == 0) {
      if (i <= first_trained_) break;
    }
    if (i > 0) {
      layers_[i-1]->ResizeDerivMat();
      layers_[i]->TransformBackward(layers_[i-1]);
    }
  }
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Backward pass finished");
  }
}

void Net::InitActiv(const MatGPU &data) {
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Initializing activations");
  }
  mexAssertMsg(layers_.size() >= 1 , "The net is not initialized");
  first_layer_ = 0;
  Layer *firstlayer = layers_[first_layer_];
  firstlayer->activ_mat_.attach(data);
  firstlayer->activ_mat_.Validate();
  if (params_.verbose_ >= 5) {
    mexPrintMsg("InitActivSum", firstlayer->activ_mat_.sum());
  }
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Activations initialized");
  }
}

void Net::InitDeriv(const MatGPU &labels_batch, const MatGPU &coef_batch, ftype &loss) {
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Initializing gradients");
  }
  size_t batchsize = labels_batch.size1();
  size_t classes_num = labels_batch.size2();
  Layer *lastlayer = layers_.back();
  ftype matsum = lastlayer->activ_mat_.sum();
  mexAssertMsg(!std::isnan(matsum), "NaNs in the network output");
  mexAssertMsg(batchsize == lastlayer->dims_[0],
    "The number of objects in data and label batches is different");
  mexAssertMsg(classes_num == lastlayer->length(),
    "Labels in batch and last layer must have equal number of classes");
  lossmat_.resize(batchsize, classes_num);
  lastlayer->deriv_mat_.resize_tensor(lastlayer->dims_);
  if (params_.lossfun_ == "logreg") {
    lossmat_ = lastlayer->activ_mat_;
    // to get the log(1) = 0 after and to avoid 0/0;
    lossmat_.CondAssign(labels_batch, false, kEps, 1);
    // to avoid log(0) and division by 0 if there are still some;
    lossmat_.CondAssign(lossmat_, false, 0, kEps);
    if (lastlayer->function_ == "soft") {
      // directly compute final derivatives, so Nonlinear is not needed
      lastlayer->deriv_mat_ = lastlayer->activ_mat_;
      lastlayer->deriv_mat_ -= labels_batch;
    } else {
      lastlayer->deriv_mat_ = labels_batch;
      (lastlayer->deriv_mat_ /= lossmat_) *= -1;
    }
    lossmat_.Log() *= -1;
  } else if (params_.lossfun_ == "L-norm") {
    lastlayer->deriv_mat_ = lastlayer->activ_mat_;
    lastlayer->deriv_mat_ -= labels_batch;
    lossmat_ = lastlayer->deriv_mat_;
    if (params_.normfun_ == 1) {
      lastlayer->deriv_mat_.Sign();
      lossmat_ *= lastlayer->deriv_mat_; // |f(x)-y|
    } else if (params_.normfun_ == 2) {
      (lossmat_ *= lastlayer->deriv_mat_) /= 2; // (f(x)-y)^2 / 2
    }
  }
  if (!coef_batch.empty()) {
    lastlayer->deriv_mat_.MultVect(coef_batch, 2);
    lossmat_.MultVect(coef_batch, 2);
  }
  lastlayer->deriv_mat_.Validate();
  loss = lossmat_.sum() / batchsize;
  if (params_.verbose_ >= 5) {
    mexPrintMsg("InitDerivSum", lastlayer->deriv_mat_.sum());
  }
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Gradients initialized");
  }
}

void Net::InitActivIBP(ftype &loss, int normfun) {
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Initializing activations for the IBP pass");
  }
  Layer *firstlayer = layers_[first_layer_];
  size_t batchsize = firstlayer->dims_[0];
  size_t length = firstlayer->length();
  lossmat2_.resize(batchsize, length);
  lossmat2_ = firstlayer->deriv_mat_;
  // fill in first_mat, because later it will be swapped with activ_mat
  firstlayer->first_mat_.resize_tensor(firstlayer->dims_);
  firstlayer->first_mat_ = firstlayer->deriv_mat_;
  if (normfun == 1) { // L1-norm
    firstlayer->first_mat_.Sign();
    lossmat2_ *= firstlayer->first_mat_; // abs
  } else if (normfun == 2) { // L2-norm
    (lossmat2_ *= firstlayer->deriv_mat_) /= 2;
  }
  loss = lossmat2_.sum() / batchsize;
  if (params_.verbose_ >= 5) {
    mexPrintMsg("InitActivIBPSum", firstlayer->first_mat_.sum());
  }
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Activations initialized");
  }
}

void Net::InitActivAT(ftype coef, int normfun) {
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Initializing activations for the AT pass");
  }
  Layer *firstlayer = layers_[first_layer_];
  if (normfun == 1) {
    firstlayer->deriv_mat_.Sign();
  }
  firstlayer->deriv_mat_ *= coef;
  firstlayer->activ_mat_ += firstlayer->deriv_mat_;
  firstlayer->activ_mat_.Validate();
  if (params_.verbose_ >= 5) {
    mexPrintMsg("InitActivATSum", firstlayer->first_mat_.sum());
  }
  if (params_.verbose_ >= 4) {
    mexPrintMsg("Activations initialized");
  }
}

void Net::UpdateWeights() {
  weights_.Update(params_);
}

void Net::ReadData(const mxArray *mx_data) {
  Dim dims = mexGetTensor(mx_data, data_);
  mexAssertMsg(layers_[0]->dims_[1] == dims[1] &&
            layers_[0]->dims_[2] == dims[2] &&
            layers_[0]->dims_[3] == dims[3],
            "Data dimensions don't correspond to the input layer");
}

void Net::ReadLabels(const mxArray *mx_labels) {
  Dim dims = mexGetTensor(mx_labels, labels_);
  mexAssertMsg(layers_.back()->dims_[1] == dims[1] &&
            layers_.back()->dims_[2] == dims[2] &&
            layers_.back()->dims_[3] == dims[3],
            "Label's dimensions don't correspond to the output layer");
  if (!classcoefs_.empty()) {
    mexAssertMsg(classcoefs_.size1() == layers_.back()->dims_[1],
      "Classcoefs vector length don't correspond to the label matrix");
  }
}

void Net::InitWeights(const mxArray *mx_weights_in) {
  if (params_.verbose_ >= 1) {
    mexPrintMsg("Start init weights");
  }
  bool isgen = false;
  size_t num_weights = NumWeights();
  MatCPU weights_cpu;
  if (mx_weights_in != NULL) { // training, testing
    if (params_.verbose_ >= 1) {
      mexPrintInt("Model weights num", num_weights);
      mexPrintInt("Input weights num", mexGetNumel(mx_weights_in));
    }
    mexAssertMsg(num_weights == mexGetNumel(mx_weights_in),
      "The vector of weights has the wrong length!");
    mexGetMatrix(mx_weights_in, weights_cpu);
  } else { // genweights
    isgen = true;
    weights_cpu.resize(1, num_weights);
  }
  weights_.Init(weights_cpu);
  size_t offset = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->InitWeights(weights_, offset, isgen);
  }
  if (params_.verbose_ >= 1) {
    mexPrintMsg("Finish init weights");
  }
}

void Net::GetWeights(mxArray *&mx_weights) const {
  size_t num_weights = NumWeights();
  mx_weights = mexNewMatrix(1, num_weights);
  MatCPU weights_cpu;
  weights_cpu.attach(mexGetPointer(mx_weights), 1, num_weights);
  size_t offset = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->RestoreOrder();
  }
  DeviceToHost(weights_.get(), weights_cpu);
}

void Net::GetLosses(mxArray *&mx_losses) const {
  mx_losses = mexSetMatrix(losses_);
}

size_t Net::NumWeights() const {
  size_t num_weights = 0;
  for (size_t i = 0; i < layers_.size(); ++i) {
    num_weights += layers_[i]->NumWeights();
    //mexPrintInt("i", num_weights);
  }
  return num_weights;
}

Net::~Net() {
  for (size_t i = 0; i < layers_.size(); ++i) {
    delete layers_[i];
  }
  layers_.clear();
  // remove here all GPU allocated memory manually,
  // otherwise CudaReset causes crash
  weights_.Clear();
  classcoefs_.clear(); // in fact vector
  lossmat_.clear();
  lossmat2_.clear();
  MatGPU::CudaReset();
}
