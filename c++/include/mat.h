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

#ifndef _MAT_H_
#define _MAT_H_

#include "settings.h"

#if COMP_REGIME != 2
  #if COMP_REGIME == 1
    #include <omp.h>    
  #endif  
  #include "mat_cpu.h"    
  typedef MatCPU Mat;
#elif COMP_REGIME == 2  
  #include "mat_gpu.h"
  typedef MatGPU Mat;    
#endif

#endif