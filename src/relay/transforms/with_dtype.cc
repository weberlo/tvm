/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file to_cps.cc
 *
 * \brief Turn a program to continuation passing style.
 *
 * Given a fresh type variable 'answer',
 * continuation passing style(CPS) convert every function of a -> b to a -> (b -> anwer) -> answer.
 *
 * That is, instead of returning the result directly,
 * function will now call another function (called the continuation)
 * and return that value as a result instead.
 *
 * Continuation passing style turn all function call into tail call,
 * which bound the stack size, prevent stack from overflowing during recursion,
 * and allow tail call optimization.
 *
 * In relay, as tensor operation is the bottleneck,
 * CPS is currently intended to transform the program before partial eval (PE),
 * as it reify the control flow and enable PE to handle control flow join more agressively.
 *
 * For example, given 'let a = if b then c else d in e', it will transform the code into
 * 'let f a = e in if b then f c else f d'.
 * This allow f to be optimized individually in both branch.
 *
 * We implement CPS conversion by higher order transform
 * (see http://matt.might.net/articles/cps-conversion/).
 * The basic idea is that we will recursively traverse the AST.
 * During the traversal, there is an extra parameter, mcont, of expr -> expr.
 * It is basically a continuation at the metalevel.
 * All cases in the transform must return via the mcont,
 * wheter directly invoking it, or indirectly by recursion.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/transform.h>

#include "let_list.h"
#include "pass_util.h"

namespace tvm {
namespace relay {

// we assume the data type has no closure - no idea how to look into datatype right now.


class DtypeReplacer : public TypeMutator {
 public:
  DtypeReplacer(DataType target_dtype)
    : target_dtype_(target_dtype) {}

  Type VisitType_(const TensorTypeNode* tt) {
    return TensorType(tt->shape, target_dtype_);
  }

 private:
  DataType target_dtype_;
};

// """Generate a type from the given type where all dtypes are replaced with the target dtype.
//
// Parameters
// ----------
// typ : relay.Type
//     Type whose dtypes are being replaced
//
// target_dtype : str
//     Target data type (e.g., 'int8')
//
// Returns
// -------
// typ : relay.Type
//     Type with only `target_dtype` for dtypes
// """
Type WithDtype(Type ty, DataType target_dtype) {
  return DtypeReplacer(target_dtype).VisitType(ty);
}

TVM_REGISTER_GLOBAL("relay._transform.with_dtype")
    .set_body_typed(WithDtype);

}  // namespace relay
}  // namespace tvm
