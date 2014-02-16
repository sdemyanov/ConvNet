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

#ifndef _MEX_PRINT_H_
#define _MEX_PRINT_H_

#include <mex.h>
#include <string>

extern int print;

inline void mexAssert(bool b, std::string msg) {
	if (!b){		
		std::string _errmsg = std::string("Assertion Failed: ") + msg;
		mexErrMsgTxt(_errmsg.c_str());
	}
}

inline void mexPrintMsg(std::string msg) {
  mexPrintf((msg + "\n").c_str());
  mexEvalString("drawnow;");
}

inline void mexPrintMsg(std::string msg, long double x) {
  mexPrintf((msg + ": " + std::to_string(x) + "\n").c_str());
  mexEvalString("drawnow;");
}

inline void mexPrintMsg(std::string msg, std::string s) {
  mexPrintf((msg + ": " + s + "\n").c_str());
  mexEvalString("drawnow;");
}

#endif