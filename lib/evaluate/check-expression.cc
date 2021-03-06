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

#include "check-expression.h"
#include "traverse.h"
#include "type.h"
#include "../semantics/symbol.h"
#include "../semantics/tools.h"

namespace Fortran::evaluate {

// Constant expression predicate IsConstantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.
class IsConstantExprHelper : public AllTraverse<IsConstantExprHelper, true> {
public:
  using Base = AllTraverse<IsConstantExprHelper, true>;
  IsConstantExprHelper() : Base{*this} {}
  using Base::operator();

  template<int KIND> bool operator()(const TypeParamInquiry<KIND> &inq) const {
    return IsKindTypeParameter(inq.parameter());
  }
  bool operator()(const semantics::Symbol &symbol) const {
    return IsNamedConstant(symbol);
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const semantics::ParamValue &param) const {
    return param.isExplicit() && (*this)(param.GetExplicit());
  }
  template<typename T> bool operator()(const FunctionRef<T> &call) const {
    if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
      return intrinsic->name == "kind";
      // TODO: other inquiry intrinsics
    } else {
      return false;
    }
  }

  // Forbid integer division by zero in constants.
  template<int KIND>
  bool operator()(
      const Divide<Type<TypeCategory::Integer, KIND>> &division) const {
    using T = Type<TypeCategory::Integer, KIND>;
    if (const auto divisor{GetScalarConstantValue<T>(division.right())}) {
      return !divisor->IsZero();
    } else {
      return false;
    }
  }
};

template<typename A> bool IsConstantExpr(const A &x) {
  return IsConstantExprHelper{}(x);
}
template bool IsConstantExpr(const Expr<SomeType> &);
template bool IsConstantExpr(const Expr<SomeInteger> &);

// Object pointer initialization checking predicate IsInitialDataTarget().
// This code determines whether an expression is allowable as the static
// data address used to initialize a pointer with "=> x".  See C765.
struct IsInitialDataTargetHelper
  : public AllTraverse<IsInitialDataTargetHelper, true> {
  using Base = AllTraverse<IsInitialDataTargetHelper, true>;
  using Base::operator();
  explicit IsInitialDataTargetHelper(parser::ContextualMessages &m)
    : Base{*this}, messages_{m} {}

  bool operator()(const BOZLiteralConstant &) const { return false; }
  bool operator()(const NullPointer &) const { return true; }
  template<typename T> bool operator()(const Constant<T> &) const {
    return false;
  }
  bool operator()(const semantics::Symbol &symbol) const {
    const Symbol &ultimate{symbol.GetUltimate()};
    if (IsAllocatable(ultimate)) {
      messages_.Say(
          "An initial data target may not be a reference to an ALLOCATABLE '%s'"_err_en_US,
          ultimate.name());
    } else if (ultimate.Corank() > 0) {
      messages_.Say(
          "An initial data target may not be a reference to a coarray '%s'"_err_en_US,
          ultimate.name());
    } else if (!ultimate.attrs().test(semantics::Attr::TARGET)) {
      messages_.Say(
          "An initial data target may not be a reference to an object '%s' that lacks the TARGET attribute"_err_en_US,
          ultimate.name());
    } else if (!IsSaved(ultimate)) {
      messages_.Say(
          "An initial data target may not be a reference to an object '%s' that lacks the SAVE attribute"_err_en_US,
          ultimate.name());
    }
    return true;
  }
  bool operator()(const StaticDataObject &) const { return false; }
  template<int KIND> bool operator()(const TypeParamInquiry<KIND> &) const {
    return false;
  }
  bool operator()(const Triplet &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        IsConstantExpr(x.stride());
  }
  bool operator()(const Subscript &x) const {
    return std::visit(
        common::visitors{
            [&](const Triplet &t) { return (*this)(t); },
            [&](const auto &y) {
              return y.value().Rank() == 0 && IsConstantExpr(y.value());
            },
        },
        x.u);
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const Substring &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        (*this)(x.parent());
  }
  bool operator()(const DescriptorInquiry &) const { return false; }
  template<typename T> bool operator()(const ArrayConstructor<T> &) const {
    return false;
  }
  bool operator()(const StructureConstructor &) const { return false; }
  template<typename T> bool operator()(const FunctionRef<T> &) { return false; }
  template<typename D, typename R, typename... O>
  bool operator()(const Operation<D, R, O...> &) const {
    return false;
  }
  template<typename T> bool operator()(const Parentheses<T> &x) const {
    return (*this)(x.left());
  }
  bool operator()(const Relational<SomeType> &) const { return false; }

private:
  parser::ContextualMessages &messages_;
};

bool IsInitialDataTarget(
    const Expr<SomeType> &x, parser::ContextualMessages &messages) {
  return IsInitialDataTargetHelper{messages}(x);
}

// Specification expression validation (10.1.11(2), C1010)
class CheckSpecificationExprHelper
  : public AnyTraverse<CheckSpecificationExprHelper,
        std::optional<std::string>> {
public:
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<CheckSpecificationExprHelper, Result>;
  explicit CheckSpecificationExprHelper(const semantics::Scope &s)
    : Base{*this}, scope_{s} {}
  using Base::operator();

  Result operator()(const ProcedureDesignator &) const {
    return "dummy procedure argument";
  }
  Result operator()(const CoarrayRef &) const { return "coindexed reference"; }

  Result operator()(const semantics::Symbol &symbol) const {
    if (semantics::IsNamedConstant(symbol)) {
      return std::nullopt;
    } else if (symbol.IsDummy()) {
      if (symbol.attrs().test(semantics::Attr::OPTIONAL)) {
        return "reference to OPTIONAL dummy argument '"s +
            symbol.name().ToString() + "'";
      } else if (symbol.attrs().test(semantics::Attr::INTENT_OUT)) {
        return "reference to INTENT(OUT) dummy argument '"s +
            symbol.name().ToString() + "'";
      } else if (symbol.has<semantics::ObjectEntityDetails>()) {
        return std::nullopt;
      } else {
        return "dummy procedure argument";
      }
    } else if (symbol.has<semantics::UseDetails>() ||
        symbol.has<semantics::HostAssocDetails>() ||
        symbol.owner().kind() == semantics::Scope::Kind::Module) {
      return std::nullopt;
    } else if (const auto *object{
                   symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      // TODO: what about EQUIVALENCE with data in COMMON?
      // TODO: does this work for blank COMMON?
      if (object->commonBlock()) {
        return std::nullopt;
      }
    }
    for (const semantics::Scope *s{&scope_}; !s->IsGlobal();) {
      s = &s->parent();
      if (s == &symbol.owner()) {
        return std::nullopt;
      }
    }
    return "reference to local entity '"s + symbol.name().ToString() + "'";
  }

  Result operator()(const Component &x) const {
    // Don't look at the component symbol.
    return (*this)(x.base());
  }
  Result operator()(const DescriptorInquiry &) const {
    // Subtle: Uses of SIZE(), LBOUND(), &c. that are valid in specification
    // expressions will have been converted to expressions over descriptor
    // inquiries by Fold().
    return std::nullopt;
  }

  template<typename T> Result operator()(const FunctionRef<T> &x) const {
    if (const auto *symbol{x.proc().GetSymbol()}) {
      if (!symbol->attrs().test(semantics::Attr::PURE)) {
        return "reference to impure function '"s + symbol->name().ToString() +
            "'";
      }
      // TODO: other checks for standard module procedures
    } else {
      const SpecificIntrinsic &intrin{DEREF(x.proc().GetSpecificIntrinsic())};
      if (intrin.name == "present") {
        return std::nullopt;  // no need to check argument(s)
      }
      if (IsConstantExpr(x)) {
        // inquiry functions may not need to check argument(s)
        return std::nullopt;
      }
    }
    return (*this)(x.arguments());
  }

private:
  const semantics::Scope &scope_;
};

template<typename A>
void CheckSpecificationExpr(const A &x, parser::ContextualMessages &messages,
    const semantics::Scope &scope) {
  if (auto why{CheckSpecificationExprHelper{scope}(x)}) {
    messages.Say("Invalid specification expression: %s"_err_en_US, *why);
  }
}

template void CheckSpecificationExpr(const Expr<SomeType> &,
    parser::ContextualMessages &, const semantics::Scope &);
template void CheckSpecificationExpr(const std::optional<Expr<SomeInteger>> &,
    parser::ContextualMessages &, const semantics::Scope &);
template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &, parser::ContextualMessages &,
    const semantics::Scope &);

// IsSimplyContiguous() -- 9.5.4
class IsSimplyContiguousHelper
  : public AnyTraverse<IsSimplyContiguousHelper, std::optional<bool>> {
public:
  using Result = std::optional<bool>;  // tri-state
  using Base = AnyTraverse<IsSimplyContiguousHelper, Result>;
  explicit IsSimplyContiguousHelper(const IntrinsicProcTable &t)
    : Base{*this}, table_{t} {}
  using Base::operator();

  Result operator()(const semantics::Symbol &symbol) const {
    if (symbol.attrs().test(semantics::Attr::CONTIGUOUS)) {
      return true;
    } else if (semantics::IsPointer(symbol)) {
      return false;
    } else if (const auto *details{
                   symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      // N.B. ALLOCATABLEs are deferred shape, not assumed, and
      // are obviously contiguous.
      return !details->IsAssumedShape() && !details->IsAssumedRank();
    } else {
      return false;
    }
  }

  Result operator()(const ArrayRef &x) const {
    if (x.base().Rank() > 0 || !CheckSubscripts(x.subscript())) {
      return false;
    } else {
      return (*this)(x.base());
    }
  }
  Result operator()(const CoarrayRef &x) const {
    return CheckSubscripts(x.subscript());
  }
  Result operator()(const Component &) const { return false; }
  Result operator()(const ComplexPart &) const { return false; }
  Result operator()(const Substring &) const { return false; }

  template<typename T> Result operator()(const FunctionRef<T> &x) const {
    if (auto chars{
            characteristics::Procedure::Characterize(x.proc(), table_)}) {
      if (chars->functionResult) {
        const auto &result{*chars->functionResult};
        return !result.IsProcedurePointer() &&
            result.attrs.test(characteristics::FunctionResult::Attr::Pointer) &&
            result.attrs.test(
                characteristics::FunctionResult::Attr::Contiguous);
      }
    }
    return false;
  }

private:
  static bool CheckSubscripts(const std::vector<Subscript> &subscript) {
    bool anyTriplet{false};
    for (auto j{subscript.size()}; j-- > 0;) {
      if (const auto *triplet{std::get_if<Triplet>(&subscript[j].u)}) {
        if (!triplet->IsStrideOne()) {
          return false;
        } else if (anyTriplet) {
          if (triplet->lower() || triplet->upper()) {
            return false;  // all triplets before the last one must be just ":"
          }
        } else {
          anyTriplet = true;
        }
      } else if (anyTriplet || subscript[j].Rank() > 0) {
        return false;
      }
    }
    return true;
  }

  const IntrinsicProcTable &table_;
};

template<typename A>
bool IsSimplyContiguous(const A &x, const IntrinsicProcTable &table) {
  if (IsVariable(x)) {
    if (auto known{IsSimplyContiguousHelper{table}(x)}) {
      return *known;
    }
  }
  return false;
}

template bool IsSimplyContiguous(
    const Expr<SomeType> &, const IntrinsicProcTable &);

}
