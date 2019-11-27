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

#include "bridge.h"
#include "ast-builder.h"
#include "builder.h"
#include "convert-expr.h"
#include "convert-type.h"
#include "flattened.h"
#include "intrinsics.h"
#include "io.h"
#include "runtime.h"
#include "../parser/parse-tree.h"
#include "../semantics/tools.h"
#include "fir/FIRDialect.h"
#include "fir/FIROps.h"
#include "fir/FIRType.h"
#include "fir/InternalNames.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"

#undef TODO
#define TODO() assert(false && "not yet implemented")

#undef SOFT_TODO
#define SOFT_TODO() \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " not yet implemented\n";

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace Fl = Fortran::burnside::flat;
namespace L = llvm;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

namespace {

using SomeExpr = Ev::Expr<Ev::SomeType>;

constexpr static bool isStopStmt(const Pa::StopStmt &stm) {
  return std::get<Pa::StopStmt::Kind>(stm.t) == Pa::StopStmt::Kind::Stop;
}

constexpr bool firLoopOp{false};

#if 0
/// Converter from Fortran to FIR
class OBSOLETE_FIRConverter {
  using LabelMapType = std::map<Fl::LabelMention, M::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  struct DoBoundsInfo {
    M::Value *doVar;

    M::Value *counter;
    M::Value *stepExpr;
    M::Operation *condition;
  };

  M::MLIRContext &mlirContext;
  const Pa::CookedSource *cooked;
  M::ModuleOp &module;
  Co::IntrinsicTypeDefaultKinds const &defaults;
  std::unique_ptr<M::OpBuilder> builder;
  LabelMapType blockMap;  // map from flattened labels to MLIR blocks
  std::list<Closure> edgeQ;
  std::map<const Pa::NonLabelDoStmt *, DoBoundsInfo> doMap;
  SymMap symbolMap;
  IntrinsicLibrary intrinsics;
  Pa::CharBlock lastKnownPos;
  bool noInsPt{false};

  inline M::OpBuilder &build() { return *builder.get(); }
  inline M::ModuleOp &getMod() { return module; }
  inline LabelMapType &blkMap() { return blockMap; }
  void setCurrentPos(const Pa::CharBlock &pos) { lastKnownPos = pos; }

  /// Construct the type of an Expr<A> expression
  M::Type exprType(const SomeExpr *expr) {
    return translateSomeExprToFIRType(&mlirContext, defaults, expr);
  }
  M::Type refExprType(const SomeExpr *expr) {
    auto type{translateSomeExprToFIRType(&mlirContext, defaults, expr)};
    return fir::ReferenceType::get(type);
  }

  M::Type getDefaultIntegerType() {
    return getFIRType(&mlirContext, defaults, IntegerCat);
  }
  M::Type getDefaultLogicalType() {
    return getFIRType(&mlirContext, defaults, LogicalCat);
  }

  M::Value *createFIRAddr(M::Location loc, const SomeExpr *expr) {
    return createSomeAddress(
        loc, build(), *expr, symbolMap, defaults, intrinsics);
  }
  M::Value *createFIRExpr(M::Location loc, const SomeExpr *expr) {
    return createSomeExpression(
        loc, build(), *expr, symbolMap, defaults, intrinsics);
  }
  M::Value *createTemp(M::Type type, Se::Symbol *symbol = nullptr) {
    return createTemporary(toLocation(), build(), symbolMap, type, symbol);
  }

  M::FuncOp genFunctionFIR(L::StringRef callee, M::FunctionType funcTy) {
    if (auto func{getNamedFunction(getMod(), callee)}) {
      return func;
    }
    return createFunction(getMod(), callee, funcTy);
  }

  M::FuncOp genRuntimeFunction(RuntimeEntryCode rec, int kind) {
    return genFunctionFIR(
        getRuntimeEntryName(rec), getRuntimeEntryType(rec, mlirContext, kind));
  }

  template<typename T> DoBoundsInfo *getBoundsInfo(const T &linearOp) {
    auto &st{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(linearOp.v->t)};
    setCurrentPos(st.source);
    auto *s{&st.statement};
    auto iter{doMap.find(s)};
    if (iter != doMap.end()) {
      return &iter->second;
    }
    assert(false && "DO context not present");
    return nullptr;
  }

  // Control flow destination
  void genFIR(bool lastWasLabel, const Fl::LabelOp &op) {
    if (lastWasLabel) {
      blkMap().insert({op.get(), build().getInsertionBlock()});
    } else {
      auto *currBlock{build().getInsertionBlock()};
      auto *newBlock{createBlock(&build())};
      blkMap().insert({op.get(), newBlock});
      if (!noInsPt) {
        build().setInsertionPointToEnd(currBlock);
        build().create<M::BranchOp>(toLocation(), newBlock);
      }
      build().setInsertionPointToStart(newBlock);
    }
  }

  // Goto statements
  void genFIR(AnalysisData &ad, const Fl::ReturnOp &op) {
    std::visit(Co::visitors{
                   [&](const Pa::ReturnStmt *stmt) { genReturnStmt(ad, stmt); },
                   [&](const auto *stmt) { genFIR(*stmt); },
               },
        op.u);
    noInsPt = true;
  }

  // IO statement with END, ERR, EOR labels
  void genFIR(const Fl::SwitchIOOp &op) {
    auto loc{toLocation(op.source)};
    (void)loc;
    TODO();
  }

  // CALL with alt-return value returned
  void genFIR(const Fl::SwitchOp &op, const Pa::CallStmt &stmt) {
    auto loc{toLocation(op.source)};
    (void)loc;
    TODO();
  }
  void genFIR(const Fl::SwitchOp &op, const Pa::ComputedGotoStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::ScalarIntExpr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    (void)e1;
    TODO();
  }
  void genFIR(const Fl::SwitchOp &op, const Pa::ArithmeticIfStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    (void)e1;
    TODO();
  }
  M::Value *fromCaseValue(const M::Location &locs, const Pa::CaseValue &val) {
    return createFIRExpr(locs, Se::GetExpr(val));
  }
  void genFIR(const Fl::SwitchOp &op, const Pa::CaseConstruct &stmt);
  void genFIR(const Fl::SwitchOp &op, const Pa::SelectRankConstruct &stmt);
  void genFIR(const Fl::SwitchOp &op, const Pa::SelectTypeConstruct &stmt);
  void genFIR(const Fl::SwitchOp &op) {
    std::visit([&](auto *construct) { genFIR(op, *construct); }, op.u);
    noInsPt = true;
  }

  void genFIR(AnalysisData &ad, const Fl::ActionOp &op);

  void pushDoContext(const Pa::NonLabelDoStmt *doStmt,
      M::Value *doVar = nullptr, M::Value *counter = nullptr,
      M::Value *stepExpr = nullptr) {
    doMap.emplace(doStmt, DoBoundsInfo{doVar, counter, stepExpr, 0});
  }

  void genLoopEnterFIR(const Pa::LoopControl::Bounds &bounds,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    auto loc{toLocation(source)};
    // evaluate e1, e2 [, e3] ...
    auto *e1{createFIRExpr(loc, Se::GetExpr(bounds.lower))};
    auto *e2{createFIRExpr(loc, Se::GetExpr(bounds.upper))};
    if (firLoopOp) {
      std::vector<M::Value *> step;
      if (bounds.step.has_value())
        step.push_back(createFIRExpr(loc, Se::GetExpr(bounds.step)));
      auto loopOp{build().create<fir::LoopOp>(loc, e1, e2, step)};
      auto *block = createBlock(&build(), &loopOp.getOperation()->getRegion(0));
      block->addArgument(M::IndexType::get(build().getContext()));
      return;
    }
    auto *nameExpr{bounds.name.thing.symbol};
    auto *name{createTemp(getDefaultIntegerType(), nameExpr)};
    M::Value *e3;
    if (bounds.step.has_value()) {
      auto *stepExpr{Se::GetExpr(bounds.step)};
      e3 = createFIRExpr(loc, stepExpr);
    } else {
      auto attr{build().getIntegerAttr(e2->getType(), 1)};
      e3 = build().create<M::ConstantOp>(loc, attr);
    }
    // name <- e1
    build().create<fir::StoreOp>(loc, e1, name);
    auto tripCounter{createTemp(getDefaultIntegerType())};
    // See 11.1.7.4.1, para. 1, item (3)
    // totalTrips ::= iteration count = a
    //   where a = (e2 - e1 + e3) / e3 if a > 0 and 0 otherwise
    auto c1{build().create<M::SubIOp>(loc, e2, e1)};
    auto c2{build().create<M::AddIOp>(loc, c1.getResult(), e3)};
    auto c3{build().create<M::DivISOp>(loc, c2.getResult(), e3)};
    auto *totalTrips{c3.getResult()};
    build().create<fir::StoreOp>(loc, totalTrips, tripCounter);
    pushDoContext(stmt, name, tripCounter, e3);
  }

  void genLoopEnterFIR(const Pa::ScalarLogicalExpr &logicalExpr,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.1, para. 2
    // See BuildLoopHeaderExpression()
    pushDoContext(stmt);
  }

  void genLoopEnterFIR(const Pa::LoopControl::Concurrent &concurrent,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.2
    TODO();
  }

  /// Lower FORALL construct (See 10.2.4)
  void genEnterFIR(const Pa::ForallConstruct &forall) {
    auto &stmt{std::get<Pa::Statement<Pa::ForallConstructStmt>>(forall.t)};
    setCurrentPos(stmt.source);
    auto &fas{stmt.statement};
    auto &ctrl{std::get<Co::Indirection<Pa::ConcurrentHeader>>(fas.t).value()};
    auto &bld{build()};
    (void)ctrl;
    (void)bld;  // FIXME
    // bld.create<fir::LoopOp>();
    for (auto &s : std::get<std::list<Pa::ForallBodyConstruct>>(forall.t)) {
      genFIROnVariant(s);
    }
    TODO();
  }
  void genFIR(const Pa::ForallAssignmentStmt &s) { genFIROnVariant(s); }

  void genExitFIR(const Pa::DoConstruct &construct) {
    if (firLoopOp) {
      build().setInsertionPointAfter(
          build().getBlock()->getParent()->getParentOp());
      return;
    }
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(construct.t)};
    setCurrentPos(stmt.source);
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<Pa::LoopControl>>(ss.t)};
    if (ctrl.has_value() &&
        std::holds_alternative<Pa::LoopControl::Bounds>(ctrl->u)) {
      doMap.erase(&ss);
    }
    noInsPt = true;  // backedge already processed
  }

  void genFIR(const Fl::EndOp &op) {
    if (auto *construct{std::get_if<const Pa::DoConstruct *>(&op.u)})
      genExitFIR(**construct);
  }

  void genFIR(AnalysisData &ad, const Fl::IndirectGotoOp &op);

  void genFIR(const Fl::DoIncrementOp &op) {
    if (firLoopOp) {
      return;
    }
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: do_var = do_var + e3
      auto load{
          build().create<fir::LoadOp>(info->doVar->getLoc(), info->doVar)};
      auto incremented{build().create<M::AddIOp>(
          load.getLoc(), load.getResult(), info->stepExpr)};
      build().create<fir::StoreOp>(load.getLoc(), incremented, info->doVar);
      // add: counter--
      auto loadCtr{
          build().create<fir::LoadOp>(info->counter->getLoc(), info->counter)};
      auto one{build().create<M::ConstantOp>(
          loadCtr.getLoc(), build().getIntegerAttr(loadCtr.getType(), 1))};
      auto decremented{build().create<M::SubIOp>(
          loadCtr.getLoc(), loadCtr.getResult(), one)};
      build().create<fir::StoreOp>(
          loadCtr.getLoc(), decremented, info->counter);
    }
  }

  void genFIR(const Fl::DoCompareOp &op) {
    if (firLoopOp) {
      return;
    }
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: cond = counter > 0 (signed)
      auto load{
          build().create<fir::LoadOp>(info->counter->getLoc(), info->counter)};
      auto zero{build().create<M::ConstantOp>(
          load.getLoc(), build().getIntegerAttr(load.getType(), 0))};
      auto cond{build().create<M::CmpIOp>(
          load.getLoc(), M::CmpIPredicate::sgt, load, zero)};
      info->condition = cond;
    }
  }

  void genReturnStmt(AnalysisData &, const Pa::FunctionSubprogram &func) {
    auto &stmt{std::get<Pa::Statement<Pa::FunctionStmt>>(func.t)};
    auto &name{std::get<Pa::Name>(stmt.statement.t)};
    assert(name.symbol);
    const auto &details{name.symbol->get<Se::SubprogramDetails>()};
    M::Value *resultRef{symbolMap.lookupSymbol(details.result())};
    assert(resultRef);  // FIXME might die if result
    // was never referenced before and temp not created.
    M::Value *resultVal{build().create<fir::LoadOp>(toLocation(), resultRef)};
    build().create<M::ReturnOp>(toLocation(), resultVal);
  }

  void genReturnStmt(const Pa::MainProgram &) {
    build().create<M::ReturnOp>(toLocation());
  }

  void genReturnStmt(
      const Pa::SubroutineSubprogram &, const Pa::ReturnStmt * = nullptr) {
    // TODO use Pa::ReturnStmt for alternate return
    build().create<M::ReturnOp>(toLocation());
  }
  void genReturnStmt(AnalysisData &ad, const Pa::ReturnStmt *stmt = nullptr) {
    std::visit(Co::visitors{
                   [&](const Pa::SubroutineSubprogram *sub) {
                     genReturnStmt(*sub, stmt);
                   },
                   [&](const Pa::FunctionSubprogram *func) {
                     genReturnStmt(ad, *func);
                   },
                   [&](const Pa::MainProgram *main) { genReturnStmt(*main); },
               },
        ad.parseTreeRoot);
  }

  // Conditional branch-like statements
  template<typename A>
  void genFIR(
      const A &tuple, Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    auto *exprRef{Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(tuple))};
    assert(exprRef && "condition expression missing");
    auto *cond{createFIRExpr(toLocation(), exprRef)};
    genCondBranch(cond, trueLabel, falseLabel);
  }
  void genFIR(const Pa::Statement<Pa::IfThenStmt> &stmt,
      Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    setCurrentPos(stmt.source);
    genFIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genFIR(const Pa::Statement<Pa::ElseIfStmt> &stmt,
      Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    setCurrentPos(stmt.source);
    genFIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genFIR(const Pa::IfStmt &stmt, Fl::LabelMention trueLabel,
      Fl::LabelMention falseLabel) {
    genFIR(stmt.t, trueLabel, falseLabel);
  }

  M::Value *getTrueConstant() {
    auto attr{build().getBoolAttr(true)};
    return build().create<M::ConstantOp>(toLocation(), attr).getResult();
  }

  // Conditional branch to enter loop body or exit
  void genFIR(const Pa::Statement<Pa::NonLabelDoStmt> &stmt,
      Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    setCurrentPos(stmt.source);
    auto &loopCtrl{std::get<std::optional<Pa::LoopControl>>(stmt.statement.t)};
    M::Value *condition{nullptr};
    bool exitNow{false};
    if (loopCtrl.has_value()) {
      exitNow =
          std::visit(Co::visitors{
                         [&](const Pa::LoopControl::Bounds &) {
                           if (firLoopOp) {
                             return true;
                           }
                           auto iter{doMap.find(&stmt.statement)};
                           assert(iter != doMap.end());
                           condition = iter->second.condition->getResult(0);
                           return false;
                         },
                         [&](const Pa::ScalarLogicalExpr &logical) {
                           auto loc{toLocation(stmt.source)};
                           auto *exp{Se::GetExpr(logical)};
                           condition = createFIRExpr(loc, exp);
                           return false;
                         },
                         [&](const Pa::LoopControl::Concurrent &concurrent) {
                           // FIXME: incorrectly lowering DO CONCURRENT
                           condition = getTrueConstant();
                           return false;
                         },
                     },
              loopCtrl->u);
      if (firLoopOp && exitNow) {
        return;
      }
    } else {
      condition = getTrueConstant();
    }
    assert(condition && "condition must be a Value");
    genCondBranch(condition, trueLabel, falseLabel);
  }

  // Action statements
  void genFIR(AnalysisData &ad, const Pa::AssignStmt &stmt) { TODO(); }

  template<typename A>
  void translateRoutine(
      const A &routine, L::StringRef name, const Se::Symbol *funcSym);

  template<typename A>
  void genSwitchBranch(const M::Location &loc, M::Value *selector,
      std::list<typename A::Conditions> &&conditions,
      const std::vector<Fl::LabelMention> &labels) {
    assert(conditions.size() == labels.size());
    bool haveAllLabels{true};
    std::size_t u{0};
    // do we already have all the targets?
    for (auto last{labels.size()}; u != last; ++u) {
      haveAllLabels = blkMap().find(labels[u]) != blkMap().end();
      if (!haveAllLabels) break;
    }
    if (haveAllLabels) {
      // yes, so generate the FIR operation now
      u = 0;
      std::vector<M::Value *> conds;
      std::vector<M::Block *> blocks;
      std::vector<L::ArrayRef<M::Value *>> blockArgs;
      L::SmallVector<M::Value *, 2> blanks;
      for (auto cond : conditions) {
        conds.emplace_back(cond);
        blocks.emplace_back(blkMap().find(labels[u++])->second);
        blockArgs.emplace_back(blanks);
      }
      build().create<A>(loc, selector, conds, blocks, blockArgs);
    } else {
      // no, so queue the FIR operation for later
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](M::OpBuilder *builder, M::Block *block, M::Value *sel,
              const std::list<typename A::Conditions> &conditions,
              const std::vector<Fl::LabelMention> &labels, M::Location location,
              const LabelMapType &map) {
            std::size_t u{0};
            std::vector<M::Value *> conds;
            std::vector<M::Block *> blocks;
            std::vector<L::ArrayRef<M::Value *>> blockArgs;
            L::SmallVector<M::Value *, 2> blanks;
            for (auto &cond : conditions) {
              auto iter{map.find(labels[u++])};
              assert(iter != map.end());
              conds.emplace_back(cond);
              blocks.emplace_back(iter->second);
              blockArgs.emplace_back(blanks);
            }
            builder->setInsertionPointToEnd(block);
            builder->create<A>(location, sel, conds, blocks, blockArgs);
          },
          &build(), build().getInsertionBlock(), selector, conditions, labels,
          loc, _1));
    }
  }

};
#endif

/// Converter from AST to FIR
///
/// After building the AST and decorating it, the FirConverter processes that
/// representation and lowers it to the FIR executable representation.
class FirConverter {
  using LabelMapType = std::map<AST::Evaluation *, M::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;
  using CFGSinkListType = L::SmallVector<AST::Evaluation *, 2>;
  using CFGMapType = L::DenseMap<AST::Evaluation *, CFGSinkListType *>;

  //
  // Function-like AST entry and exit statements
  //

  void genFIR(const Pa::Statement<Pa::ProgramStmt> &stmt, std::string &name,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    name = mangler.getProgramEntry();
  }
  void genFIR(const Pa::Statement<Pa::EndProgramStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::FunctionStmt> &stmt, std::string &name,
      Se::Symbol const *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{std::get<Pa::Name>(stmt.statement.t)};
    name = n.ToString();
    symbol = n.symbol;
  }
  void genFIR(const Pa::Statement<Pa::EndFunctionStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::SubroutineStmt> &stmt, std::string &name,
      Se::Symbol const *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{std::get<Pa::Name>(stmt.statement.t)};
    name = n.ToString();
    symbol = n.symbol;
  }
  void genFIR(const Pa::Statement<Pa::EndSubroutineStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::MpSubprogramStmt> &stmt,
      std::string &name, Se::Symbol const *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{stmt.statement.v};
    name = n.ToString();
    symbol = n.symbol;
  }
  void genFIR(const Pa::Statement<Pa::EndMpSubprogramStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }

  //
  // Termination of symbolically referenced execution units
  //

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genFIRProgramExit() { SOFT_TODO(); }
  void genFIR(const Pa::EndProgramStmt &) { genFIRProgramExit(); }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genFIRProcedureExit() { SOFT_TODO(); }
  void genFIR(const Pa::EndFunctionStmt &) { genFIRProcedureExit(); }
  void genFIR(const Pa::EndSubroutineStmt &) { genFIRProcedureExit(); }
  void genFIR(const Pa::EndMpSubprogramStmt &) { genFIRProcedureExit(); }

  //
  // Statements that have control-flow semantics
  //

  // Conditional goto control-flow semantics
  void genFIREvalCondGoto(AST::Evaluation &eval) {
    genFIR(eval);
    auto targets{findTargetsOf(eval)};
    // FIXME - thread condition
    genFIRCondBranch(nullptr, targets[0], targets[1]);
  }

  void genFIRCondBranch(
      M::Value *cond, AST::Evaluation *trueDest, AST::Evaluation *falseDest) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](M::OpBuilder *builder, M::Block *block, M::Value *cnd,
            AST::Evaluation *trueDest, AST::Evaluation *falseDest,
            M::Location location, const LabelMapType &map) {
          L::SmallVector<M::Value *, 2> blk;
          builder->setInsertionPointToEnd(block);
          auto tdp{map.find(trueDest)};
          auto fdp{map.find(falseDest)};
          assert(tdp != map.end() && fdp != map.end());
          builder->create<M::CondBranchOp>(
              location, cnd, tdp->second, blk, fdp->second, blk);
        },
        builder, builder->getInsertionBlock(), cond, trueDest, falseDest,
        toLocation(), _1));
  }

  // Goto control-flow semantics
  //
  // These are unconditional jumps. There is nothing to evaluate.
  void genFIREvalGoto(AST::Evaluation &eval) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](M::OpBuilder *builder, M::Block *block, AST::Evaluation *dest,
            M::Location location, const LabelMapType &map) {
          builder->setInsertionPointToEnd(block);
          assert(map.find(dest) != map.end() && "no destination");
          builder->create<M::BranchOp>(location, map.find(dest)->second);
        },
        builder, builder->getInsertionBlock(), findSinkOf(eval), toLocation(),
        _1));
  }

  // Indirect goto control-flow semantics
  //
  // For assigned gotos, which is an obsolescent feature. Lower to a switch.
  void genFIREvalIndGoto(AST::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // IO statements that have control-flow semantics
  //
  // First lower the IO statement and then do the multiway switch op
  void genFIREvalIoSwitch(AST::Evaluation &eval) {
    genFIR(eval);
    genFIRIOSwitch(eval);
  }
  void genFIRIOSwitch(AST::Evaluation &) { TODO(); }

  // Iterative loop control-flow semantics
  void genFIREvalIterative(AST::Evaluation &eval) {
    // FIXME
  }

  // Return from subprogram control-flow semantics
  void genFIREvalReturn(AST::Evaluation &eval) {
    genFIR(eval);
    M::Value *resultVal = nullptr;  // FIXME
    builder->create<M::ReturnOp>(toLocation(), resultVal);
  }

  // Multiway switch control-flow semantics
  void genFIREvalSwitch(AST::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // Terminate process control-flow semantics
  //
  // Call a runtime routine that does not return
  void genFIREvalTerminate(AST::Evaluation &eval) {
    genFIR(eval);
    builder->create<fir::UnreachableOp>(toLocation());
  }

  // No control-flow
  void genFIREvalNone(AST::Evaluation &eval) { genFIR(eval); }

  void genFIR(const Pa::CallStmt &) { TODO(); }
  void genFIR(const Pa::IfStmt &) { TODO(); }
  void genFIR(const Pa::WaitStmt &) { TODO(); }
  void genFIR(const Pa::WhereStmt &) { TODO(); }
  void genFIR(const Pa::ComputedGotoStmt &) { TODO(); }
  void genFIR(const Pa::ForallStmt &) { TODO(); }
  void genFIR(const Pa::ArithmeticIfStmt &) { TODO(); }
  void genFIR(const Pa::AssignedGotoStmt &) { TODO(); }

  void genFIR(const Pa::AssociateConstruct &) { TODO(); }
  void genFIR(const Pa::BlockConstruct &) { TODO(); }
  void genFIR(const Pa::CaseConstruct &) { TODO(); }
  void genFIR(const Pa::ChangeTeamConstruct &) { TODO(); }
  void genFIR(const Pa::CriticalConstruct &) { TODO(); }
  void genFIR(const Pa::DoConstruct &d) {
#if 0
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(d.t)};
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<Pa::LoopControl>>(ss.t)};
    if (ctrl.has_value()) {
      std::visit([&](const auto &x) { genLoopEnterFIR(x, &ss, stmt.source); },
          ctrl->u);
    } else {
      // loop forever (See 11.1.7.4.1, para. 2)
      pushDoContext(&ss);
    }
#endif
  }
  void genFIR(const Pa::IfConstruct &cst) { TODO(); }
  void genFIR(const Pa::SelectRankConstruct &) { TODO(); }
  void genFIR(const Pa::SelectTypeConstruct &) { TODO(); }
  void genFIR(const Pa::WhereConstruct &) { TODO(); }
  void genFIR(const Pa::ForallConstruct &f) {
    // genEnterFIR(f);
  }
  void genFIR(const Pa::CompilerDirective &) { TODO(); }
  void genFIR(const Pa::OpenMPConstruct &) { TODO(); }
  void genFIR(const Pa::OmpEndLoopDirective &) { TODO(); }

  void genFIR(const parser::AssociateStmt &) { TODO(); }
  void genFIR(const parser::EndAssociateStmt &) { TODO(); }
  void genFIR(const parser::BlockStmt &) { TODO(); }
  void genFIR(const parser::EndBlockStmt &) { TODO(); }
  void genFIR(const parser::SelectCaseStmt &) { TODO(); }
  void genFIR(const parser::CaseStmt &) { TODO(); }
  void genFIR(const parser::EndSelectStmt &) { TODO(); }
  void genFIR(const parser::ChangeTeamStmt &) { TODO(); }
  void genFIR(const parser::EndChangeTeamStmt &) { TODO(); }
  void genFIR(const parser::CriticalStmt &) { TODO(); }
  void genFIR(const parser::EndCriticalStmt &) { TODO(); }
  void genFIR(const parser::NonLabelDoStmt &) { TODO(); }
  void genFIR(const parser::EndDoStmt &) { TODO(); }
  void genFIR(const parser::IfThenStmt &) { TODO(); }
  void genFIR(const parser::ElseIfStmt &) { TODO(); }
  void genFIR(const parser::ElseStmt &) { TODO(); }
  void genFIR(const parser::EndIfStmt &) { TODO(); }
  void genFIR(const parser::SelectRankStmt &) { TODO(); }
  void genFIR(const parser::SelectRankCaseStmt &) { TODO(); }
  void genFIR(const parser::SelectTypeStmt &) { TODO(); }
  void genFIR(const parser::TypeGuardStmt &) { TODO(); }
  void genFIR(const parser::WhereConstructStmt &) { TODO(); }
  void genFIR(const parser::MaskedElsewhereStmt &) { TODO(); }
  void genFIR(const parser::ElsewhereStmt &) { TODO(); }
  void genFIR(const parser::EndWhereStmt &) { TODO(); }
  void genFIR(const parser::ForallConstructStmt &) { TODO(); }
  void genFIR(const parser::EndForallStmt &) { TODO(); }

  //
  // Statements that do not have control-flow semantics
  //

  void genFIR(const Pa::AllocateStmt &) { TODO(); }
  void genFIR(const Pa::AssignmentStmt &stmt) {
#if 0
    auto *rhs{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *lhs{Se::GetExpr(std::get<Pa::Variable>(stmt.t))};
    auto loc{toLocation()};
    build().create<fir::StoreOp>(
        loc, createFIRExpr(loc, rhs), createFIRAddr(loc, lhs));
#endif
  }
  void genFIR(const Pa::BackspaceStmt &) { TODO(); }

  void genFIR(const Pa::CloseStmt &) { TODO(); }
  void genFIR(const Pa::ContinueStmt &) { TODO(); }
  void genFIR(const Pa::DeallocateStmt &) { TODO(); }
  void genFIR(const Pa::EndfileStmt &) { TODO(); }
  void genFIR(const Pa::EventPostStmt &) { TODO(); }
  void genFIR(const Pa::EventWaitStmt &) { TODO(); }
  void genFIR(const Pa::FlushStmt &) { TODO(); }
  void genFIR(const Pa::FormTeamStmt &) { TODO(); }
  void genFIR(const Pa::InquireStmt &) { TODO(); }
  void genFIR(const Pa::LockStmt &) { TODO(); }
  void genFIR(const Pa::NullifyStmt &) { TODO(); }
  void genFIR(const Pa::OpenStmt &) { TODO(); }
  void genFIR(const Pa::PointerAssignmentStmt &) { TODO(); }
  void genFIR(const Pa::PrintStmt &stmt) {
#if 0
    L::SmallVector<mlir::Value *, 4> args;
    for (const Pa::OutputItem &item :
        std::get<std::list<Pa::OutputItem>>(stmt.t)) {
      if (const Pa::Expr * parserExpr{std::get_if<Pa::Expr>(&item.u)}) {
        mlir::Location loc{toLocation(parserExpr->source)};
        args.push_back(createFIRExpr(loc, Se::GetExpr(*parserExpr)));
      } else {
        TODO();  // implied do
      }
    }
    genPrintStatement(build(), toLocation(lastKnownPos), args);
#endif
  }
  void genFIR(const Pa::ReadStmt &) { TODO(); }
  void genFIR(const Pa::RewindStmt &) { TODO(); }
  void genFIR(const Pa::SyncAllStmt &) { TODO(); }
  void genFIR(const Pa::SyncImagesStmt &) { TODO(); }
  void genFIR(const Pa::SyncMemoryStmt &) { TODO(); }
  void genFIR(const Pa::SyncTeamStmt &) { TODO(); }
  void genFIR(const Pa::UnlockStmt &) { TODO(); }

  void genFIR(const Pa::WriteStmt &) { TODO(); }
  void genFIR(const Pa::AssignStmt &) { TODO(); }
  void genFIR(const Pa::FormatStmt &) { TODO(); }
  void genFIR(const Pa::EntryStmt &) { TODO(); }
  void genFIR(const Pa::PauseStmt &) { TODO(); }
  void genFIR(const Pa::DataStmt &) { TODO(); }
  void genFIR(const Pa::NamelistStmt &) { TODO(); }

  // call FAIL IMAGE in runtime
  void genFIR(const Pa::FailImageStmt &stmt) {
#if 0
    auto callee{genRuntimeFunction(FIRT_FAIL_IMAGE, 0)};
    L::SmallVector<M::Value *, 1> operands;  // FAIL IMAGE has no args
    build().create<M::CallOp>(toLocation(), callee, operands);
#endif
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(const Pa::StopStmt &stm) {
#if 0
    auto callee{
        genRuntimeFunction(isStopStmt(stm) ? FIRT_STOP : FIRT_ERROR_STOP,
            defaults.GetDefaultKind(IntegerCat))};
    // 2 args: stop-code-opt, quiet-opt
    L::SmallVector<M::Value *, 8> operands;
    build().create<M::CallOp>(toLocation(), callee, operands);
#endif
  }

  // gen expression, if any
  void genFIR(const Pa::ReturnStmt &) { TODO(); }

  // stubs for generic goto statements; see genFIREvalGoto
  void genFIR(const Pa::CycleStmt &) { assert(false && "invalid"); }
  void genFIR(const Pa::ExitStmt &) { assert(false && "invalid"); }
  void genFIR(const Pa::GotoStmt &) { assert(false && "invalid"); }

  void genFIR(AST::Evaluation &eval) {
    std::visit(Co::visitors{
                   [&](const auto *p) { genFIR(*p); },
                   [](const AST::CGJump &) { assert(false && "invalid"); },
               },
        eval.u);
  }

  void lowerEval(AST::Evaluation &eval) {
    setCurrentPosition(eval.pos);
    if (eval.isControlTarget()) {
      // start a new block
    }
    switch (eval.cfg) {
    case AST::CFGAnnotation::None: genFIREvalNone(eval); break;
    case AST::CFGAnnotation::Goto: genFIREvalGoto(eval); break;
    case AST::CFGAnnotation::CondGoto: genFIREvalCondGoto(eval); break;
    case AST::CFGAnnotation::IndGoto: genFIREvalIndGoto(eval); break;
    case AST::CFGAnnotation::IoSwitch: genFIREvalIoSwitch(eval); break;
    case AST::CFGAnnotation::Switch: genFIREvalSwitch(eval); break;
    case AST::CFGAnnotation::Iterative: genFIREvalIterative(eval); break;
    case AST::CFGAnnotation::Return: genFIREvalReturn(eval); break;
    case AST::CFGAnnotation::Terminate: genFIREvalTerminate(eval); break;
    }
  }

  M::FuncOp createNewFunction(L::StringRef name, const Se::Symbol *symbol) {
    // get arguments and return type if any, otherwise just use empty vectors
    L::SmallVector<M::Type, 8> args;
    L::SmallVector<M::Type, 2> results;
    if (symbol) {
      auto *details{symbol->detailsIf<Se::SubprogramDetails>()};
      assert(details && "details for semantics::Symbol must be subprogram");
      for (auto *a : details->dummyArgs()) {
        if (a) {  // nullptr indicates alternate return argument
          auto type{translateSymbolToFIRType(&mlirContext, defaults, *a)};
          args.push_back(fir::ReferenceType::get(type));
        }
      }
      if (details->isFunction()) {
        // FIXME: handle subroutines that return magic values
        auto result{details->result()};
        results.push_back(
            translateSymbolToFIRType(&mlirContext, defaults, result));
      }
    }
    auto funcTy{M::FunctionType::get(args, results, &mlirContext)};
    return createFunction(module, name, funcTy);
  }

  /// Prepare to translate a new function
  void startNewFunction(L::StringRef name, const Se::Symbol *symbol) {
    M::FuncOp func{getNamedFunction(module, name)};
    if (!func) {
      func = createNewFunction(name, symbol);
    }
    func.addEntryBlock();
    assert(!builder && "expected nullptr");
    builder = new M::OpBuilder(func);
    assert(builder && "OpBuilder did not instantiate");
    builder->setInsertionPointToStart(&func.front());

    // plumb function's arguments
    if (symbol) {
      auto *entryBlock{&func.front()};
      auto *details{symbol->detailsIf<Se::SubprogramDetails>()};
      assert(details && "details for semantics::Symbol must be subprogram");
      for (const auto &v :
          L::zip(details->dummyArgs(), entryBlock->getArguments())) {
        if (std::get<0>(v)) {
          localSymbols.addSymbol(*std::get<0>(v), std::get<1>(v));
        } else {
          TODO();  // handle alternate return
        }
      }
    }
  }

  void finalizeQueuedEdges() {
    for (auto &edgeFunc : localEdgeQ) {
      edgeFunc(localBlockMap);
    }
    localEdgeQ.clear();
    localBlockMap.clear();
  }

  /// Cleanup after the function has been translated
  void endNewFunction() {
    finalizeQueuedEdges();
    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Lower a procedure-like construct
  void lowerFunc(AST::FunctionLikeUnit &func, L::ArrayRef<L::StringRef> modules,
      L::Optional<L::StringRef> host = {}) {
    std::string name;
    const Se::Symbol *symbol{nullptr};
    auto size{func.funStmts.size()};

    assert((size == 1 || size == 2) && "ill-formed subprogram");
    if (size == 2) {
      std::visit(
          [&](auto *p) { genFIR(*p, name, symbol); }, func.funStmts.front());
    } else {
      name = mangler.getProgramEntry();
    }

    startNewFunction(name, symbol);

    // lower this procedure
    for (auto &e : func.evals) {
      lowerEval(e);
    }
    std::visit(
        [&](auto *p) { genFIR(*p, name, symbol); }, func.funStmts.back());

    endNewFunction();

    // recursively lower internal procedures
    L::Optional<L::StringRef> optName{name};
    for (auto &f : func.funcs) {
      lowerFunc(f, modules, optName);
    }
  }

  void lowerMod(AST::ModuleLikeUnit &mod) {
    // FIXME: build the vector of module names
    std::vector<L::StringRef> moduleName;

    // FIXME: do we need to visit the module statements?
    for (auto &f : mod.funcs) {
      lowerFunc(f, moduleName);
    }
  }

  //
  // Finalization of the CFG structure
  //

  /// Lookup the set of sinks for this source. There must be at least one.
  L::ArrayRef<AST::Evaluation *> findTargetsOf(AST::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert(iter != cfgMap.end());
    return *iter->second;
  }

  /// Lookup the sink for this source. There must be exactly one.
  AST::Evaluation *findSinkOf(AST::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert((iter != cfgMap.end()) && (iter->second->size() == 1));
    return iter->second->front();
  }

  /// Add source->sink edge to CFG map
  void addSourceToSink(AST::Evaluation *src, AST::Evaluation *snk) {
    auto iter = cfgMap.find(src);
    if (iter == cfgMap.end()) {
      CFGSinkListType sink{snk};
      cfgEdgeSetPool.emplace_back(std::move(sink));
      auto rc{cfgMap.try_emplace(src, &cfgEdgeSetPool.back())};
      assert(rc.second && "insert failed unexpectedly");
      return;
    }
    for (auto *s : *iter->second)
      if (s == snk) {
        return;
      }
    iter->second->push_back(snk);
  }

  /// prune the CFG for `f`
  void pruneFunc(AST::FunctionLikeUnit &func) {
    // find and cache arcs, etc.
    SOFT_TODO();

    // do any internal procedures
    for (auto &f : func.funcs) {
      pruneFunc(f);
    }
  }

  void pruneMod(AST::ModuleLikeUnit &mod) {
    for (auto &f : mod.funcs) {
      pruneFunc(f);
    }
  }

  void setCurrentPosition(const Pa::CharBlock &pos) {
    if (pos != Pa::CharBlock{}) {
      currentPosition = pos;
    }
  }

  //
  // Utility methods
  //

  /// Convert a parser CharBlock to a Location
  M::Location toLocation(const Pa::CharBlock &cb) {
    return parserPosToLoc(mlirContext, cooked, cb);
  }

  M::Location toLocation() { return toLocation(currentPosition); }

  // TODO: should these be moved to convert-expr?
  template<M::CmpIPredicate ICMPOPC>
  M::Value *genCompare(M::Value *lhs, M::Value *rhs) {
    auto lty{lhs->getType()};
    assert(lty == rhs->getType());
    if (lty.isIntOrIndex()) {
      return builder->create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (fir::LogicalType::kindof(lty.getKind())) {
      return builder->create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (fir::CharacterType::kindof(lty.getKind())) {
      // FIXME
      // return builder->create<M::CallOp>(lhs->getLoc(), );
    }
    assert(false && "cannot generate operation on this type");
    return {};
  }
  M::Value *genGE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::sge>(lhs, rhs);
  }
  M::Value *genLE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::sle>(lhs, rhs);
  }
  M::Value *genEQ(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::eq>(lhs, rhs);
  }
  M::Value *genAND(M::Value *lhs, M::Value *rhs) {
    return builder->create<M::AndOp>(lhs->getLoc(), lhs, rhs);
  }

private:
  M::MLIRContext &mlirContext;
  const Pa::CookedSource *cooked;
  M::ModuleOp &module;
  Co::IntrinsicTypeDefaultKinds const &defaults;
  IntrinsicLibrary intrinsics;
  M::OpBuilder *builder{nullptr};
  fir::NameMangler &mangler;
  SymMap localSymbols;
  std::list<Closure> localEdgeQ;
  LabelMapType localBlockMap;
  Pa::CharBlock currentPosition;
  CFGMapType cfgMap;
  std::list<CFGSinkListType> cfgEdgeSetPool;

public:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  explicit FirConverter(BurnsideBridge &bridge, fir::NameMangler &mangler)
    : mlirContext{bridge.getMLIRContext()}, cooked{bridge.getCookedSource()},
      module{bridge.getModule()}, defaults{bridge.getDefaultKinds()},
      intrinsics{IntrinsicLibrary::create(
          IntrinsicLibrary::Version::LLVM, bridge.getMLIRContext())},
      mangler{mangler} {}

  /// Convert the AST to FIR
  void run(AST::Program &ast) {
    // build pruned control
    for (auto &u : ast.getUnits()) {
      std::visit(common::visitors{
                     [&](AST::FunctionLikeUnit &f) { pruneFunc(f); },
                     [&](AST::ModuleLikeUnit &m) { pruneMod(m); },
                     [](AST::BlockDataUnit &) { /* do nothing */ },
                 },
          u);
    }

    // do translation
    for (auto &u : ast.getUnits()) {
      std::visit(common::visitors{
                     [&](AST::FunctionLikeUnit &f) { lowerFunc(f, {}); },
                     [&](AST::ModuleLikeUnit &m) { lowerMod(m); },
                     [](AST::BlockDataUnit &) { /* FIXME */ },
                 },
          u);
    }
  }
};

#if 0
/// SELECT CASE
/// Build a switch-like structure for a SELECT CASE
void OBSOLETE_FIRConverter::genFIR(
    const Fl::SwitchOp &op, const Pa::CaseConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &cstm{std::get<Pa::Statement<Pa::SelectCaseStmt>>(stmt.t)};
  auto *exp{Se::GetExpr(std::get<Pa::Scalar<Pa::Expr>>(cstm.statement.t))};
  auto *e1{createFIRExpr(loc, exp)};
  auto &cases{std::get<std::list<Pa::CaseConstruct::Case>>(stmt.t)};
  std::list<fir::SelectCaseOp::Conditions> conds;
  // Per C1145, we know each `case-expr` must have type INTEGER, CHARACTER, or
  // LOGICAL
  for (auto &sel : cases) {
    auto &cs{std::get<Pa::Statement<Pa::CaseStmt>>(sel.t)};
    auto locs{toLocation(cs.source)};
    auto &csel{std::get<Pa::CaseSelector>(cs.statement.t)};
    std::visit(
        Co::visitors{
            [&](const std::list<Pa::CaseValueRange> &ranges) {
              for (auto &r : ranges) {
                std::visit(Co::visitors{
                               [&](const Pa::CaseValue &val) {
                                 auto *term{fromCaseValue(locs, val)};
                                 conds.emplace_back(genEQ(e1, term));
                               },
                               [&](const Pa::CaseValueRange::Range &rng) {
                                 fir::SelectCaseOp::Conditions rangeComparison =
                                     nullptr;
                                 if (rng.lower.has_value()) {
                                   auto *term{fromCaseValue(locs, *rng.lower)};
                                   // rc = e1 >= lower.term
                                   rangeComparison = genGE(e1, term);
                                 }
                                 if (rng.upper.has_value()) {
                                   auto *term{fromCaseValue(locs, *rng.upper)};
                                   // c = e1 <= upper.term
                                   auto *comparison{genLE(e1, term)};
                                   // rc = if rc then (rc && c) else c
                                   if (rangeComparison) {
                                     rangeComparison =
                                         genAND(rangeComparison, comparison);
                                   } else {
                                     rangeComparison = comparison;
                                   }
                                 }
                                 conds.emplace_back(rangeComparison);
                               },
                           },
                    r.u);
              }
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        csel.u);
  }
  genSwitchBranch<fir::SelectCaseOp>(loc, e1, std::move(conds), op.refs);
}

/// SELECT RANK
/// Build a switch-like structure for a SELECT RANK
void OBSOLETE_FIRConverter::genFIR(
    const Fl::SwitchOp &op, const Pa::SelectRankConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &rstm{std::get<Pa::Statement<Pa::SelectRankStmt>>(stmt.t)};
  auto *exp{std::visit([](auto &x) { return Se::GetExpr(x); },
      std::get<Pa::Selector>(rstm.statement.t).u)};
  auto *e1{createFIRExpr(loc, exp)};
  auto &ranks{std::get<std::list<Pa::SelectRankConstruct::RankCase>>(stmt.t)};
  std::list<fir::SelectRankOp::Conditions> conds;
  for (auto &r : ranks) {
    auto &rs{std::get<Pa::Statement<Pa::SelectRankCaseStmt>>(r.t)};
    auto &rank{std::get<Pa::SelectRankCaseStmt::Rank>(rs.statement.t)};
    std::visit(
        Co::visitors{
            [&](const Pa::ScalarIntConstantExpr &ex) {
              auto *ie{createFIRExpr(loc, Se::GetExpr(ex))};
              conds.emplace_back(ie);
            },
            [&](const Pa::Star &) {
              // FIXME: using a bogon for now.  Special value per
              // whatever the runtime returns.
              auto attr{build().getIntegerAttr(e1->getType(), -1)};
              conds.emplace_back(build().create<M::ConstantOp>(loc, attr));
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        rank.u);
  }
  // FIXME: fix the type of the function
  auto callee{genRuntimeFunction(FIRT_GET_RANK, 0)};
  L::SmallVector<M::Value *, 1> operands{e1};
  auto e3{build().create<M::CallOp>(loc, callee, operands)};
  genSwitchBranch<fir::SelectRankOp>(
      loc, e3.getResult(0), std::move(conds), op.refs);
}

/// SELECT TYPE
/// Build a switch-like structure for a SELECT TYPE
void OBSOLETE_FIRConverter::genFIR(
    const Fl::SwitchOp &op, const Pa::SelectTypeConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &tstm{std::get<Pa::Statement<Pa::SelectTypeStmt>>(stmt.t)};
  auto *exp{std::visit([](auto &x) { return Se::GetExpr(x); },
      std::get<Pa::Selector>(tstm.statement.t).u)};
  auto *e1{createFIRExpr(loc, exp)};
  auto &types{std::get<std::list<Pa::SelectTypeConstruct::TypeCase>>(stmt.t)};
  std::list<fir::SelectTypeOp::Conditions> conds;
  for (auto &t : types) {
    auto &ts{std::get<Pa::Statement<Pa::TypeGuardStmt>>(t.t)};
    auto &ty{std::get<Pa::TypeGuardStmt::Guard>(ts.statement.t)};
    std::visit(
        Co::visitors{
            [&](const Pa::TypeSpec &) {
              // FIXME: add arguments
              auto func{genRuntimeFunction(FIRT_ISA_TYPE, 0)};
              L::SmallVector<M::Value *, 2> operands;
              auto call{build().create<M::CallOp>(loc, func, operands)};
              conds.emplace_back(call.getResult(0));
            },
            [&](const Pa::DerivedTypeSpec &) {
              // FIXME: add arguments
              auto func{genRuntimeFunction(FIRT_ISA_SUBTYPE, 0)};
              L::SmallVector<M::Value *, 2> operands;
              auto call{build().create<M::CallOp>(loc, func, operands)};
              conds.emplace_back(call.getResult(0));
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        ty.u);
  }
  auto callee{genRuntimeFunction(FIRT_GET_ELETYPE, 0)};
  L::SmallVector<M::Value *, 1> operands{e1};
  auto e3{build().create<M::CallOp>(loc, callee, operands)};
  genSwitchBranch<fir::SelectTypeOp>(
      loc, e3.getResult(0), std::move(conds), op.refs);
}

/// translate action statements
void OBSOLETE_FIRConverter::genFIR(AnalysisData &ad, const Fl::ActionOp &op) {
  setCurrentPos(op.v->source);
  std::visit(Co::visitors{
                 [](const Pa::ContinueStmt &) { TODO(); },
                 [](const Pa::FailImageStmt &) { TODO(); },
                 [](const Co::Indirection<Pa::ArithmeticIfStmt> &) { TODO(); },
                 [](const Co::Indirection<Pa::AssignedGotoStmt> &) { TODO(); },
                 [](const Co::Indirection<Pa::ComputedGotoStmt> &) { TODO(); },
                 [](const Co::Indirection<Pa::CycleStmt> &) { TODO(); },
                 [](const Co::Indirection<Pa::ExitStmt> &) { TODO(); },
                 [](const Co::Indirection<Pa::GotoStmt> &) { TODO(); },
                 [](const Co::Indirection<Pa::IfStmt> &) { TODO(); },
                 [](const Co::Indirection<Pa::StopStmt> &) { TODO(); },
                 [&](const Co::Indirection<Pa::AssignStmt> &assign) {
                   genFIR(ad, assign.value());
                 },
                 [](const Co::Indirection<Pa::ReturnStmt> &) {
                   assert(false && "should be a ReturnOp");
                 },
                 [&](const auto &stmt) { genFIR(stmt); },
             },
      op.v->statement.u);
}

void OBSOLETE_FIRConverter::genFIR(AnalysisData &ad, const Fl::IndirectGotoOp &op) {
  // add or queue an igoto
  TODO();
}

void OBSOLETE_FIRConverter::genFIR(AnalysisData &ad, std::list<Fl::Op> &operations) {
  bool lastWasLabel{false};
  for (auto &op : operations) {
    std::visit(Co::visitors{
                   [&](const Fl::IndirectGotoOp &oper) {
                     genFIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Fl::ActionOp &oper) {
                     noInsPt = false;
                     genFIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Fl::LabelOp &oper) {
                     genFIR(lastWasLabel, oper);
                     lastWasLabel = true;
                   },
                   [&](const Fl::BeginOp &oper) {
                     noInsPt = false;
                     genFIR(oper);
                     lastWasLabel = true;
                   },
                   [&](const Fl::ReturnOp &oper) {
                     noInsPt = false;
                     genFIR(ad, oper);
                     lastWasLabel = true;
                   },
                   [&](const auto &oper) {
                     noInsPt = false;
                     genFIR(oper);
                     lastWasLabel = false;
                   },
               },
        op.u);
  }
  if (build().getInsertionBlock()) {
    genReturnStmt(ad);
  }
}
#endif

}  // namespace

void Br::BurnsideBridge::lower(
    const Pa::Program &prg, fir::NameMangler &mangler) {
  AST::Program ast{Br::createAST(prg)};
  Br::annotateControl(ast);
  FirConverter converter{*this, mangler};
  converter.run(ast);
}

void Br::BurnsideBridge::parseSourceFile(L::SourceMgr &srcMgr) {
  auto owningRef = M::parseSourceFile(srcMgr, context.get());
  module.reset(new M::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Br::BurnsideBridge::BurnsideBridge(
    const Co::IntrinsicTypeDefaultKinds &defaultKinds,
    const Pa::CookedSource *cooked)
  : defaultKinds{defaultKinds}, cooked{cooked} {
  context = std::make_unique<M::MLIRContext>();
  module = std::make_unique<M::ModuleOp>(
      M::ModuleOp::create(M::UnknownLoc::get(context.get())));
}
