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

enum class CFGAnnotation {
  None,
  Goto,
  CondGoto,
  IndGoto,
  IoSwitch,
  Switch,
  Return
};

class StatementLike {};

/// A Part is the variant of parser objects that a Evaluation points to
using Part = std::variant<const parser::AllocateStmt *,
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
    const parser::AssignedGotoStmt *, const parser::PauseStmt *,
    const parser::FormatStmt *, const parser::EntryStmt *,
    const parser::DataStmt *, const parser::NamelistStmt *,
    const parser::AssociateConstruct *, const parser::BlockConstruct *,
    const parser::CaseConstruct *, const parser::ChangeTeamConstruct *,
    const parser::CriticalConstruct *, const parser::DoConstruct *,
    const parser::IfConstruct *, const parser::SelectRankConstruct *,
    const parser::SelectTypeConstruct *, const parser::WhereConstruct *,
    const parser::ForallConstruct *, const parser::CompilerDirective *,
    const parser::OpenMPConstruct *, const parser::OmpEndLoopDirective *>;

/// AST Statement: This flattens out a parse tree Statement (gets rid of
/// indirections and grammar production structure
template<typename A, typename B> struct Statement {
  using TYPE = StatementLike;

  Statement() = delete;
  Statement(const A &a, const parser::Statement<B> &b) : p{&a}, s{&b} {}

  bool possibleBranchTarget() const { return s->label.has_value(); }
  parser::CharBlock getCBLocation() const { return s->source; }

  const A *p{nullptr};
  const parser::Statement<B> *s{nullptr};
};

template<typename A>
struct IndirectStmt : public Statement<A, common::Indirection<A>> {
  IndirectStmt() = delete;
  IndirectStmt(const parser::Statement<common::Indirection<A>> &stmt)
    : Statement<A, common::Indirection<A>>{stmt.statement.value(), stmt} {}
};

template<typename A>
struct ActionStmt : public Statement<A, parser::ActionStmt> {
  ActionStmt() = delete;
  ActionStmt(const A &ptr, const parser::Statement<parser::ActionStmt> &stmt)
    : Statement<A, parser::ActionStmt>{ptr, stmt} {}
};

struct Evaluation;
class ConstructLike {};

/// AST Construct which can contain a list of evaluations
template<typename A> struct Construct {
  using TYPE = ConstructLike;

  Construct() = delete;
  Construct(const A &ptr) : p{&ptr} {}

  const A *p;
  std::list<Evaluation> evals;
};

/// Function-like units can contains lists of evaluations.  These can be
/// (simple) statements or constructs, where a construct contains its own
/// evaluations.
struct Evaluation {
  Evaluation() = delete;
  template<typename A> Evaluation(const A &a) : u{a} {}

  void setCFG(CFGAnnotation a) { cfg = a; }

  constexpr bool isConstruct() const {
    return std::visit(
        [](const auto &x) {
          using T = typename std::decay<decltype(x)>::type::TYPE;
          return std::is_same<ConstructLike, T>::value;
        },
        u);
  }

  constexpr bool isStatement() const {
    return std::visit(
        [](const auto &x) {
          using T = typename std::decay<decltype(x)>::type::TYPE;
          return std::is_same<StatementLike, T>::value;
        },
        u);
  }

  std::list<Evaluation> *getConstructEvals();
  Part getPart();

  std::variant<ActionStmt<parser::AllocateStmt>,
      ActionStmt<parser::AssignmentStmt>, ActionStmt<parser::BackspaceStmt>,
      ActionStmt<parser::CallStmt>, ActionStmt<parser::CloseStmt>,
      ActionStmt<parser::ContinueStmt>, ActionStmt<parser::CycleStmt>,
      ActionStmt<parser::DeallocateStmt>, ActionStmt<parser::EndfileStmt>,
      ActionStmt<parser::EventPostStmt>, ActionStmt<parser::EventWaitStmt>,
      ActionStmt<parser::ExitStmt>, ActionStmt<parser::FailImageStmt>,
      ActionStmt<parser::FlushStmt>, ActionStmt<parser::FormTeamStmt>,
      ActionStmt<parser::GotoStmt>, ActionStmt<parser::IfStmt>,
      ActionStmt<parser::InquireStmt>, ActionStmt<parser::LockStmt>,
      ActionStmt<parser::NullifyStmt>, ActionStmt<parser::OpenStmt>,
      ActionStmt<parser::PointerAssignmentStmt>, ActionStmt<parser::PrintStmt>,
      ActionStmt<parser::ReadStmt>, ActionStmt<parser::ReturnStmt>,
      ActionStmt<parser::RewindStmt>, ActionStmt<parser::StopStmt>,
      ActionStmt<parser::SyncAllStmt>, ActionStmt<parser::SyncImagesStmt>,
      ActionStmt<parser::SyncMemoryStmt>, ActionStmt<parser::SyncTeamStmt>,
      ActionStmt<parser::UnlockStmt>, ActionStmt<parser::WaitStmt>,
      ActionStmt<parser::WhereStmt>, ActionStmt<parser::WriteStmt>,
      ActionStmt<parser::ComputedGotoStmt>, ActionStmt<parser::ForallStmt>,
      ActionStmt<parser::ArithmeticIfStmt>, ActionStmt<parser::AssignStmt>,
      ActionStmt<parser::AssignedGotoStmt>, ActionStmt<parser::PauseStmt>,

      IndirectStmt<parser::FormatStmt>, IndirectStmt<parser::EntryStmt>,
      IndirectStmt<parser::DataStmt>, IndirectStmt<parser::NamelistStmt>,

      Construct<parser::AssociateConstruct>, Construct<parser::BlockConstruct>,
      Construct<parser::CaseConstruct>, Construct<parser::ChangeTeamConstruct>,
      Construct<parser::CriticalConstruct>, Construct<parser::DoConstruct>,
      Construct<parser::IfConstruct>, Construct<parser::SelectRankConstruct>,
      Construct<parser::SelectTypeConstruct>, Construct<parser::WhereConstruct>,
      Construct<parser::ForallConstruct>, Construct<parser::CompilerDirective>,
      Construct<parser::OpenMPConstruct>,
      Construct<parser::OmpEndLoopDirective>>
      u;
  CFGAnnotation cfg{CFGAnnotation::None};
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

  const semantics::Scope *scope{nullptr};
  std::list<FunctionStatement> funStmts;
  std::list<Evaluation> evals;
  std::list<FunctionLikeUnit> funcs;
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
  std::list<Units> units;
};

}  // namespace AST

/// Create an AST from the parse tree
AST::Program createAST(const parser::Program &root);

/// Decorate the AST with control flow annotations
void annotateControl(AST::Program &ast);

}  // namespace burnside

#endif  // FORTRAN_BURNSIDE_AST_BUILDER_H_
