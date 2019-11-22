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

#include "ast-builder.h"
#include "../parser/parse-tree-visitor.h"
#include <cassert>
#include <utility>

/// Build an light-weight AST to help with lowering to FIR.  The AST will
/// capture pointers back into the parse tree, so the parse tree data structure
/// may <em>not</em> be changed between the construction of the AST and all of
/// its uses.
///
/// The AST captures a structured view of the program.  The program is a list of
/// units.  Function like units will contain lists of evaluations.  Evaluations
/// are either statements or constructs, where a construct contains a list of
/// evaluations.  The resulting AST structure can then be used to create FIR.

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Pa = Fortran::parser;

using namespace Fortran;
using namespace Br;

namespace {

/// The instantiation of a parse tree visitor (Pre and Post) is extremely
/// expensive in terms of compile and link time, so one goal here is to limit
/// the bridge to one such instantiation.
class ASTBuilder {
public:
  ASTBuilder() = default;

  /// Get the result
  AST::Program result() { return pgm; }

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  // Module like

  bool Pre(const Pa::Module &x) { return enterModule(x); }
  bool Pre(const Pa::Submodule &x) { return enterModule(x); }

  void Post(const Pa::Module &) { exitModule(); }
  void Post(const Pa::Submodule &) { exitModule(); }

  // Function like

  bool Pre(const Pa::MainProgram &x) { return enterFunc(x); }
  bool Pre(const Pa::FunctionSubprogram &x) { return enterFunc(x); }
  bool Pre(const Pa::SubroutineSubprogram &x) { return enterFunc(x); }
  bool Pre(const Pa::SeparateModuleSubprogram &x) { return enterFunc(x); }

  void Post(const Pa::MainProgram &) { exitFunc(); }
  void Post(const Pa::FunctionSubprogram &) { exitFunc(); }
  void Post(const Pa::SubroutineSubprogram &) { exitFunc(); }
  void Post(const Pa::SeparateModuleSubprogram &) { exitFunc(); }

  // Block data

  void Post(const Pa::BlockData &x) {
    AST::BlockDataUnit unit{x};
    addUnit(unit);
  }

  // Evaluation

  void Post(const Pa::Statement<Pa::ActionStmt> &s) {
    addEvaluation(makeEvalAction(s));
  }
  void Post(const Pa::Statement<Co::Indirection<Pa::FormatStmt>> &s) {
    addEval(AST::IndirectStmt{s});
  }
  void Post(const Pa::Statement<Co::Indirection<Pa::EntryStmt>> &s) {
    addEval(AST::IndirectStmt{s});
  }
  void Post(const Pa::Statement<Co::Indirection<Pa::DataStmt>> &s) {
    addEval(AST::IndirectStmt{s});
  }
  void Post(const Pa::Statement<Co::Indirection<Pa::NamelistStmt>> &s) {
    addEval(AST::IndirectStmt{s});
  }

  bool Pre(const Pa::AssociateConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::BlockConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::CaseConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::ChangeTeamConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::CriticalConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::DoConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::IfConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::SelectRankConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::SelectTypeConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::WhereConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::ForallConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::CompilerDirective &c) { return enterConstruct(c); }
  bool Pre(const Pa::OpenMPConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::OmpEndLoopDirective &c) { return enterConstruct(c); }

  void Post(const Pa::AssociateConstruct &) { exitConstruct(); }
  void Post(const Pa::BlockConstruct &) { exitConstruct(); }
  void Post(const Pa::CaseConstruct &) { exitConstruct(); }
  void Post(const Pa::ChangeTeamConstruct &) { exitConstruct(); }
  void Post(const Pa::CriticalConstruct &) { exitConstruct(); }
  void Post(const Pa::DoConstruct &) { exitConstruct(); }
  void Post(const Pa::IfConstruct &) { exitConstruct(); }
  void Post(const Pa::SelectRankConstruct &) { exitConstruct(); }
  void Post(const Pa::SelectTypeConstruct &) { exitConstruct(); }
  void Post(const Pa::WhereConstruct &) { exitConstruct(); }
  void Post(const Pa::ForallConstruct &) { exitConstruct(); }
  void Post(const Pa::CompilerDirective &) { exitConstruct(); }
  void Post(const Pa::OpenMPConstruct &) { exitConstruct(); }
  void Post(const Pa::OmpEndLoopDirective &) { exitConstruct(); }

private:
  // ActionStmt has a couple of non-conforming cases, which get handled
  // explicitly here.  The other cases use an Indirection, which we discard in
  // the AST.
  AST::Evaluation makeEvalAction(const Pa::Statement<Pa::ActionStmt> &s) {
    return std::visit(
        common::visitors{
            [&](const Pa::ContinueStmt &x) {
              return AST::Evaluation{AST::ActionStmt{x, s}};
            },
            [&](const Pa::FailImageStmt &x) {
              return AST::Evaluation{AST::ActionStmt{x, s}};
            },
            [&](const auto &x) {
              return AST::Evaluation{AST::ActionStmt{x.value(), s}};
            },
        },
        s.statement.u);
  }

  // When we enter a function-like structure, we want to build a new unit and
  // set the builder's cursors to point to it.
  template<typename A> bool enterFunc(const A &f) {
    AST::FunctionLikeUnit unit{f};
    funclist = &unit.funcs;
    pushEval(&unit.evals);
    addFunc(unit);
    return true;
  }

  void exitFunc() {
    funclist = nullptr;
    popEval();
  }

  // When we enter a construct structure, we want to build a new construct and
  // set the builder's evaluation cursor to point to it.
  template<typename A> bool enterConstruct(const A &c) {
    AST::Construct con{c};
    addEval(con);
    pushEval(&con.evals);
    return true;
  }

  void exitConstruct() { popEval(); }

  // When we enter a module structure, we want to build a new module and
  // set the builder's function cursor to point to it.
  template<typename A> bool enterModule(const A &f) {
    AST::ModuleLikeUnit unit{f};
    funclist = &unit.funcs;
    addUnit(unit);
    return true;
  }

  void exitModule() { funclist = nullptr; }

  template<typename A> void addUnit(const A &unit) {
    pgm.units.emplace_back(unit);
  }

  template<typename A> void addFunc(const A &func) {
    if (funclist)
      funclist->emplace_back(func);
    else
      addUnit(func);
  }

  template<typename A> void addEval(const A &eval) {
    addEvaluation(AST::Evaluation{eval});
  }

  void addEvaluation(const AST::Evaluation &eval) {
    assert(funclist);
    evallist.back()->emplace_back(eval);
  }
  void pushEval(std::list<AST::Evaluation> *eval) {
    assert(funclist);
    evallist.push_back(eval);
  }

  void popEval() {
    assert(funclist);
    evallist.pop_back();
  }

  AST::Program pgm;
  std::list<AST::FunctionLikeUnit> *funclist{nullptr};
  std::vector<std::list<AST::Evaluation> *> evallist;
};

template<typename A> void ioLabel(AST::Evaluation &e, const A *s) {
  // FIXME
}

void annotateEvalListCFG(std::list<AST::Evaluation> &evals) {
  for (auto e : evals) {
    if (e.isConstruct()) {
      annotateEvalListCFG(*e.getConstructEvals());
      continue;
    }
    std::visit(
        common::visitors{
            [&](Pa::BackspaceStmt *s) { ioLabel(e, s); },
            [&](Pa::CallStmt *) {},
            [&](Pa::CloseStmt *s) { ioLabel(e, s); },
            [&](Pa::CycleStmt *) { e.setCFG(AST::CFGAnnotation::Goto); },
            [&](Pa::EndfileStmt *s) { ioLabel(e, s); },
            [&](Pa::ExitStmt *) { e.setCFG(AST::CFGAnnotation::Goto); },
            [&](Pa::FailImageStmt *) { e.setCFG(AST::CFGAnnotation::Return); },
            [&](Pa::FlushStmt *s) { ioLabel(e, s); },
            [&](Pa::GotoStmt *) { e.setCFG(AST::CFGAnnotation::Goto); },
            [&](Pa::IfStmt *) { e.setCFG(AST::CFGAnnotation::CondGoto); },
            [&](Pa::InquireStmt *s) { ioLabel(e, s); },
            [&](Pa::OpenStmt *s) { ioLabel(e, s); },
            [&](Pa::ReadStmt *s) { ioLabel(e, s); },
            [&](Pa::ReturnStmt *) { e.setCFG(AST::CFGAnnotation::Return); },
            [&](Pa::RewindStmt *s) { ioLabel(e, s); },
            [&](Pa::StopStmt *) { e.setCFG(AST::CFGAnnotation::Return); },
            [&](Pa::WaitStmt *s) { ioLabel(e, s); },
            [&](Pa::WriteStmt *s) { ioLabel(e, s); },
            [&](Pa::ArithmeticIfStmt *) {
              e.setCFG(AST::CFGAnnotation::Switch);
            },
            [&](Pa::AssignedGotoStmt *) {
              e.setCFG(AST::CFGAnnotation::IndGoto);
            },
            [&](Pa::ComputedGotoStmt *) {
              e.setCFG(AST::CFGAnnotation::Switch);
            },
            [](auto *) {},
        },
        e.getPart());
  }
}

inline void annotateFuncCFG(AST::FunctionLikeUnit &flu) {
  annotateEvalListCFG(flu.evals);
}

}  // namespace

std::list<AST::Evaluation> *Br::AST::Evaluation::getConstructEvals() {
  return std::visit(
      [](auto &ct) -> std::list<AST::Evaluation> * {
        using T = typename std::decay<decltype(ct)>::type::TYPE;
        if constexpr (std::is_same<AST::ConstructLike, T>::value) {
          return &ct.evals;
        } else {
          return nullptr;
        }
      },
      u);
}

AST::Part Br::AST::Evaluation::getPart() {
  return std::visit([](auto &x) { return AST::Part{x.p}; }, u);
}

Br::AST::FunctionLikeUnit::FunctionLikeUnit(const Pa::MainProgram &f)
  : ProgramUnit{&f} {
  auto &ps{std::get<std::optional<Pa::Statement<Pa::ProgramStmt>>>(f.t)};
  if (ps.has_value()) {
    const Pa::Statement<Pa::ProgramStmt> &s{ps.value()};
    funStmts.push_back(&s);
  }
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndProgramStmt>>(f.t));
}

Br::AST::FunctionLikeUnit::FunctionLikeUnit(const Pa::FunctionSubprogram &f)
  : ProgramUnit{&f} {
  funStmts.push_back(&std::get<Pa::Statement<Pa::FunctionStmt>>(f.t));
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndFunctionStmt>>(f.t));
}

Br::AST::FunctionLikeUnit::FunctionLikeUnit(const Pa::SubroutineSubprogram &f)
  : ProgramUnit{&f} {
  funStmts.push_back(&std::get<Pa::Statement<Pa::SubroutineStmt>>(f.t));
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndSubroutineStmt>>(f.t));
}

Br::AST::FunctionLikeUnit::FunctionLikeUnit(
    const Pa::SeparateModuleSubprogram &f)
  : ProgramUnit{&f} {
  funStmts.push_back(&std::get<Pa::Statement<Pa::MpSubprogramStmt>>(f.t));
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndMpSubprogramStmt>>(f.t));
}

Br::AST::ModuleLikeUnit::ModuleLikeUnit(const Pa::Module &m) : ProgramUnit{&m} {
  modStmts.push_back(&std::get<Pa::Statement<Pa::ModuleStmt>>(m.t));
  modStmts.push_back(&std::get<Pa::Statement<Pa::EndModuleStmt>>(m.t));
}

Br::AST::ModuleLikeUnit::ModuleLikeUnit(const Pa::Submodule &m)
  : ProgramUnit{&m} {
  modStmts.push_back(&std::get<Pa::Statement<Pa::SubmoduleStmt>>(m.t));
  modStmts.push_back(&std::get<Pa::Statement<Pa::EndSubmoduleStmt>>(m.t));
}

Br::AST::BlockDataUnit::BlockDataUnit(const Pa::BlockData &db)
  : ProgramUnit{&db} {}

AST::Program Br::createAST(const Pa::Program &root) {
  ASTBuilder walker;
  Walk(root, walker);
  return walker.result();
}

void Br::annotateControl(AST::Program &ast) {
  for (auto unit : ast.units) {
    std::visit(common::visitors{
                   [](AST::BlockDataUnit &) {},
                   [](AST::FunctionLikeUnit &f) {
                     annotateFuncCFG(f);
                     for (auto s : f.funcs) {
                       annotateFuncCFG(s);
                     }
                   },
                   [](AST::ModuleLikeUnit &u) {
                     for (auto f : u.funcs) {
                       annotateFuncCFG(f);
                     }
                   },
               },
        unit);
  }
}
