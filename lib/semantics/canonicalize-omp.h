// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_SEMANTICS_CANONICALIZE_OMP_H_
#define FORTRAN_SEMANTICS_CANONICALIZE_OMP_H_

namespace Fortran::parser {
struct Program;
class Messages;
}

namespace Fortran::semantics {
bool CanonicalizeOmp(parser::Messages &messages, parser::Program &program);
}

#endif  // FORTRAN_SEMANTICS_CANONICALIZE_OMP_H_
