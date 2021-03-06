#if 0 /* this header can be included into both Fortran and C */
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file defines various code values that need to be exported
to predefined Fortran standard modules as well as to C/C++
code in the compiler and runtime library.
These include:
 - the error/end code values that can be returned
   to an IOSTAT= or STAT= specifier on a Fortran I/O statement
   or coindexed data reference (see Fortran 2018 12.11.5,
   16.10.2, and 16.10.2.33)
#endif
#ifndef FORTRAN_RUNTIME_MAGIC_NUMBERS_H_
#define FORTRAN_RUNTIME_MAGIC_NUMBERS_H_

#define FORTRAN_RUNTIME_IOSTAT_END (-1)
#define FORTRAN_RUNTIME_IOSTAT_EOR (-2)
#define FORTRAN_RUNTIME_IOSTAT_FLUSH (-3)
#define FORTRAN_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT 1

#define FORTRAN_RUNTIME_STAT_FAILED_IMAGE 10
#define FORTRAN_RUNTIME_STAT_LOCKED 11
#define FORTRAN_RUNTIME_STAT_LOCKED_OTHER_IMAGE 12
#define FORTRAN_RUNTIME_STAT_STOPPED_IMAGE 13
#define FORTRAN_RUNTIME_STAT_UNLOCKED 14
#define FORTRAN_RUNTIME_STAT_UNLOCKED_FAILED_IMAGE 15
#endif
