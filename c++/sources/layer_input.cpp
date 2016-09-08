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

#include "layer_input.h"

LayerInput::LayerInput() {
  function_ = "none";
  add_bias_ = false;
  dims_[1] = 1;
}

void LayerInput::Init(const mxArray *mx_layer, const Layer *prev_layer) {
  mexAssertMsg(prev_layer == NULL, "The 'input' type layer must be the first one");
}
