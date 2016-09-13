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

#ifndef _MEX_PRINT_H_
#define _MEX_PRINT_H_

#include <mex.h>
#include <string>

extern int print;

inline void mexPrintMsg(std::string msg) {
  mexPrintf((msg + "\n").c_str());
  mexEvalString("drawnow;");
}

inline void mexPrintMsg(std::string msg, double x) {
  mexPrintf((msg + ": " + std::to_string((long double) x) + "\n").c_str());
  mexEvalString("drawnow;");
}

inline void mexPrintMsg(std::string msg, std::string s) {
  mexPrintf((msg + ": " + s + "\n").c_str());
  mexEvalString("drawnow;");
}

inline void mexPrintInt(std::string msg, size_t x) {
  mexPrintf((msg + ": " + std::to_string((long long) x) + "\n").c_str());
  mexEvalString("drawnow;");
}

inline void _assertFunction(bool cond, std::string msg, const char *file, int line) {
  if (!(cond)) {
    if (!msg.empty()) {
      mexPrintf((msg + "\n").c_str());
    }
    mexPrintf((std::string(file) + ": " + std::to_string(line) + "\n").c_str());
    mexEvalString("drawnow;");
    mexErrMsgTxt("Assertion Failed!");
  }
}

#ifndef mexAssert
#define mexAssert(cond) { _assertFunction((cond), "", __FILE__, __LINE__); }
#endif

#ifndef mexAssertMsg
#define mexAssertMsg(cond, msg) { _assertFunction((cond), (msg), __FILE__, __LINE__); }
#endif

#endif
