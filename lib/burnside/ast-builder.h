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

#ifndef FORTRAN_BURNSIDE_AST_BUILDER_H_
#define FORTRAN_BURNSIDE_AST_BUILDER_H_

#include "../parser/parse-tree.h"
#include "../semantics/scope.h"

namespace Fortran::burnside {
namespace AST {

struct Evaluation;

enum class CFGAnnotation {
  None,
  Goto,
  CondGoto,
  IndGoto,
  IoSwitch,
  Switch,
  Return,
  Terminate
};

/// Compiler-generated jump
///
/// This is used to convert implicit control-flow edges to explicit form in the
/// decorated AST
struct CGJump {
  CGJump(Evaluation *to) : target{to} {}
  Evaluation *target{nullptr};
};

/// Function-like units can contains lists of evaluations.  These can be
/// (simple) statements or constructs, where a construct contains its own
/// evaluations.
struct Evaluation {
  using EvalVariant = std::variant<const parser::AllocateStmt *,
      const parser::AssignmentStmt *, const parser::BackspaceStmt *,
      const parser::CallStmt *, const parser::CloseStmt *,
      const parser::ContinueStmt *, const parser::CycleStmt *,
      const parser::DeallocateStmt *, const parser::EndfileStmt *,
      const parser::EventPostStmt *, const parser::EventWaitStmt *,
      const parser::ExitStmt *, const parser::FailImageStmt *,
      const parser::FlushStmt *, const parser::FormTeamStmt *,
      const parser::GotoStmt *, const parser::IfStmt *,
      const parser::InquireStmt *, const parser::LockStmt *,
      const parser::NullifyStmt *, const parser::OpenStmt *,
      const parser::PointerAssignmentStmt *, const parser::PrintStmt *,
      const parser::ReadStmt *, const parser::ReturnStmt *,
      const parser::RewindStmt *, const parser::StopStmt *,
      const parser::SyncAllStmt *, const parser::SyncImagesStmt *,
      const parser::SyncMemoryStmt *, const parser::SyncTeamStmt *,
      const parser::UnlockStmt *, const parser::WaitStmt *,
      const parser::WhereStmt *, const parser::WriteStmt *,
      const parser::ComputedGotoStmt *, const parser::ForallStmt *,
      const parser::ArithmeticIfStmt *, const parser::AssignStmt *,
      const parser::AssignedGotoStmt *, const parser::PauseStmt *, CGJump,
      const parser::FormatStmt *, const parser::EntryStmt *,
      const parser::DataStmt *, const parser::NamelistStmt *,
      const parser::AssociateConstruct *, const parser::BlockConstruct *,
      const parser::CaseConstruct *, const parser::ChangeTeamConstruct *,
      const parser::CriticalConstruct *, const parser::DoConstruct *,
      const parser::IfConstruct *, const parser::SelectRankConstruct *,
      const parser::SelectTypeConstruct *, const parser::WhereConstruct *,
      const parser::ForallConstruct *, const parser::CompilerDirective *,
      const parser::OpenMPConstruct *, const parser::OmpEndLoopDirective *>;

  Evaluation() = delete;
  Evaluation(const Evaluation &) = default;

  /// General ctor
  template<typename A>
  Evaluation(const A &a, const parser::CharBlock &pos,
      const std::optional<parser::Label> &lab)
    : u{&a}, pos{pos}, lab{lab} {}

  /// Compiler-generated jump
  Evaluation(const CGJump &jump) : u{jump}, cfg{CFGAnnotation::Goto} {}

  /// Construct ctor
  template<typename A> Evaluation(const A &a) : u{&a} {
    static_assert(isConstruct(a) && "must be a construct");
  }

  /// is `A` executable (an action statement or compiler generated)?
  template<typename A> constexpr static bool isAction(const A &a) {
    return !isConstruct(a) && !isOther(a);
  }

  /// is `A` a compiler-generated evaluation?
  template<typename A> constexpr static bool isGenerated(const A &) {
    return std::is_same_v<A, CGJump>;
  }

  /// is `A` a construct (or directive)?
  template<typename A> constexpr static bool isConstruct(const A &) {
    return std::is_same_v<A, parser::AssociateConstruct> ||
        std::is_same_v<A, parser::BlockConstruct> ||
        std::is_same_v<A, parser::CaseConstruct> ||
        std::is_same_v<A, parser::ChangeTeamConstruct> ||
        std::is_same_v<A, parser::CriticalConstruct> ||
        std::is_same_v<A, parser::DoConstruct> ||
        std::is_same_v<A, parser::IfConstruct> ||
        std::is_same_v<A, parser::SelectRankConstruct> ||
        std::is_same_v<A, parser::SelectTypeConstruct> ||
        std::is_same_v<A, parser::WhereConstruct> ||
        std::is_same_v<A, parser::ForallConstruct> ||
        std::is_same_v<A, parser::CompilerDirective> ||
        std::is_same_v<A, parser::OpenMPConstruct> ||
        std::is_same_v<A, parser::OmpEndLoopDirective>;
  }

  /// is `A` not an executable statement?
  template<typename A> constexpr static bool isOther(const A &) {
    return std::is_same_v<A, parser::FormatStmt> ||
        std::is_same_v<A, parser::EntryStmt> ||
        std::is_same_v<A, parser::DataStmt> ||
        std::is_same_v<A, parser::NamelistStmt>;
  }

  constexpr bool isActionStmt() const {
    return std::visit(common::visitors{
                          [](auto *p) { return isAction(*p); },
                          [](auto &r) { return isGenerated(r); },
                      },
        u);
  }

  constexpr bool isStmt() const {
    return std::visit(common::visitors{
                          [](auto *p) { return isAction(*p) || isOther(*p); },
                          [](auto &r) { return isGenerated(r); },
                      },
        u);
  }
  constexpr bool isConstruct() const { return !isStmt(); }

  /// Set the type of originating control flow type for this evaluation.
  void setCFG(CFGAnnotation a, Evaluation *cstr) {
    cfg = a;
    setBranches(cstr);
  }

  /// Is this evaluation a control-flow origin? (The AST must be annotated)
  bool isControlOrigin() const { return cfg != CFGAnnotation::None; }

  /// Is this evaluation a control-flow target? (The AST must be annotated)
  bool isControlTarget() const { return isTarget; }

  /// Set the containsBranches flag iff this evaluation (a construct) contains
  /// control flow
  void setBranches() { containsBranches = true; }

  constexpr std::list<Evaluation> *getConstructEvals() {
    return isStmt() ? nullptr : subs;
  }

  /// Set that the construct `cstr` (if not a nullptr) has branches.
  static void setBranches(Evaluation *cstr) {
    if (cstr) {
      cstr->setBranches();
    }
  }

  EvalVariant u;
  parser::CharBlock pos;
  std::optional<parser::Label> lab;
  std::list<Evaluation> *subs{nullptr};  // construct sub-statements
  CFGAnnotation cfg{CFGAnnotation::None};
  bool isTarget{false};  // this evaluation is a control target
  bool containsBranches{false};  // construct contains branches
};

/// A program is a list of program units.
/// These units can be function like, module like, or block data
struct ProgramUnit {
  template<typename A> ProgramUnit(A *ptr) : p{ptr} {}

  std::variant<const parser::MainProgram *, const parser::FunctionSubprogram *,
      const parser::SubroutineSubprogram *, const parser::Module *,
      const parser::Submodule *, const parser::SeparateModuleSubprogram *,
      const parser::BlockData *>
      p;
};

/// Function-like units have similar structure. They all can contain executable
/// statements.
struct FunctionLikeUnit : public ProgramUnit {
  // wrapper statements for function-like syntactic structures
  using FunctionStatement =
      std::variant<const parser::Statement<parser::ProgramStmt> *,
          const parser::Statement<parser::EndProgramStmt> *,
          const parser::Statement<parser::FunctionStmt> *,
          const parser::Statement<parser::EndFunctionStmt> *,
          const parser::Statement<parser::SubroutineStmt> *,
          const parser::Statement<parser::EndSubroutineStmt> *,
          const parser::Statement<parser::MpSubprogramStmt> *,
          const parser::Statement<parser::EndMpSubprogramStmt> *>;

  FunctionLikeUnit(const parser::MainProgram &f);
  FunctionLikeUnit(const parser::FunctionSubprogram &f);
  FunctionLikeUnit(const parser::SubroutineSubprogram &f);
  FunctionLikeUnit(const parser::SeparateModuleSubprogram &f);

  const semantics::Scope *scope{nullptr};  // scope from front-end
  std::list<FunctionStatement> funStmts;  // begin/end pair
  std::list<Evaluation> evals;  // statements
  std::list<FunctionLikeUnit> funcs;  // internal procedures
};

/// Module-like units have similar structure. They all can contain a list of
/// function-like units.
struct ModuleLikeUnit : public ProgramUnit {
  // wrapper statements for module-like syntactic structures
  using ModuleStatement =
      std::variant<const parser::Statement<parser::ModuleStmt> *,
          const parser::Statement<parser::EndModuleStmt> *,
          const parser::Statement<parser::SubmoduleStmt> *,
          const parser::Statement<parser::EndSubmoduleStmt> *>;

  ModuleLikeUnit(const parser::Module &m);
  ModuleLikeUnit(const parser::Submodule &m);
  ~ModuleLikeUnit() = default;

  const semantics::Scope *scope{nullptr};
  std::list<ModuleStatement> modStmts;
  std::list<FunctionLikeUnit> funcs;
};

struct BlockDataUnit : public ProgramUnit {
  BlockDataUnit(const parser::BlockData &db);
};

/// A Program is the top-level AST
struct Program {
  using Units = std::variant<FunctionLikeUnit, ModuleLikeUnit, BlockDataUnit>;

  std::list<Units> &getUnits() { return units; }

private:
  std::list<Units> units;
};

}  // namespace AST

/// Create an AST from the parse tree
AST::Program createAST(const parser::Program &root);

/// Decorate the AST with control flow annotations
///
/// The AST must be decorated with control-flow annotations to prepare it for
/// use in generating a CFG-like structure.
void annotateControl(AST::Program &ast);

}  // namespace burnside

#endif  // FORTRAN_BURNSIDE_AST_BUILDER_H_
