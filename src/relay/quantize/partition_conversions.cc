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
 * \file partition_conversions.cc
 */
#include <set>
#include <tuple>

#include <tvm/relay/analysis.h>
// #include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
// #include <tvm/relay/op.h>
// #include <tvm/relay/transform.h>
// TODO how do we *absolutely* include this?
#include "../transforms/let_list.h"

// #include "pattern_util.h"

namespace tvm {
namespace relay {
namespace quantize {

// class InferenceSimplifier : public ExprMutator {
//  public:
//   InferenceSimplifier()
//       : batch_norm_op_(Op::Get("nn.batch_norm")),
//         dropout_op_(Op::Get("nn.dropout")),
//         instance_norm_op_(Op::Get("nn.instance_norm")),
//         layer_norm_op_(Op::Get("nn.layer_norm")),
//         group_norm_op_(Op::Get("nn.group_norm")),
//         l2_norm_op_(Op::Get("nn.l2_normalize")) {}

//   Expr VisitExpr_(const TupleGetItemNode* n) final {
//     Expr new_e = ExprMutator::VisitExpr_(n);
//     const auto* new_n = new_e.as<TupleGetItemNode>();
//     if (new_n->index != 0) {
//       return new_e;
//     }
//     if (const auto* call = new_n->tuple.as<CallNode>()) {
//       if (call->op == batch_norm_op_) {
//         return BatchNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
//                                       call->args[3], call->args[4], ty_map_.at(call->args[0]));
//       } else if (call->op == dropout_op_) {
//         return call->args[0];
//       }
//     }
//     return new_e;
//   }

//   Expr VisitExpr_(const CallNode* n) {
//     auto new_n = ExprMutator::VisitExpr_(n);
//     if (n->op == batch_norm_op_) {
//       ty_map_[new_n.as<CallNode>()->args[0]] = n->args[0]->checked_type();
//     } else if (n->op == layer_norm_op_) {
//       const auto* call = new_n.as<CallNode>();
//       return LayerNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
//                                     n->args[0]->checked_type());
//     } else if (n->op == group_norm_op_) {
//       const auto* call = new_n.as<CallNode>();
//       return GroupNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
//                                     n->args[0]->checked_type());
//     } else if (n->op == instance_norm_op_) {
//       const auto* call = new_n.as<CallNode>();
//       return InstanceNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
//                                        n->args[0]->checked_type());
//     } else if (n->op == l2_norm_op_) {
//       const auto* call = new_n.as<CallNode>();
//       return L2NormToInferUnpack(call->attrs, call->args[0]);
//     }
//     return new_n;
//   }
//  private:
//   const Op& batch_norm_op_;
//   std::unordered_map<Expr, Type, ObjectPtrHash, ObjectPtrEqual> ty_map_;
// };

// operators that are allowed in prefix/suffix partitions, because they are used
// to quantize/dequantize
std::set<std::string> kAllowedConversionOps = {
  "add", "multiply", "right_shift", "clip", "round", "cast"
};

Type WithDtype(Type ty, DataType dtype) {
  CHECK(false) << "unimpled";
  return ty;
}

// A mutator for extracting input quantization expressions from a function
//
// The result of `visit` is the core function, and the input quantization
// expressions are stored in the `prefix_sb` scope builder.
class PrefixCutter : public ExprMutator {
 public:
  PrefixCutter(Array<relay::Var> params, std::set<DLDataType> quantized_dtypes)
    : quantized_dtypes_(quantized_dtypes) {
    for (const auto p : params) {
      params_.insert(p);
    }
  }

  Expr VisitExpr_(const VarNode* op) final {
    const auto var = GetRef<Var>(op);
    if (params_.find(var) != params_.end()) {
      subtree_params_.insert(var);
    }
    return var;
  }

  Expr VisitExpr_(const CallNode* op) final {
    const auto call = GetRef<Call>(op);
    const auto* call_op = call->op.as<OpNode>();
    if (call_op && kAllowedConversionOps.find(call_op->name) != kAllowedConversionOps.end()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Array<Expr> new_args;
      for (const auto& arg : call->args) {
        const Expr& new_arg = VisitExpr(arg);
        if (subtree_params_.size() == 0) {
          new_args.push_back(new_arg);
        } else {
          CHECK_EQ(subtree_params_.size(), 1)
            << "found multiple parameters at base of quantization conversion subexpression";
          const Var& param = *(subtree_params_.begin());
          const Var pre_param = prefix_ll_.Push(Var(param->name_hint(), Type()), new_arg);
          subtree_params_.clear();
          const auto* tt = arg->checked_type().as<TensorTypeNode>();
          const Var mid_param = Var(
            param->name_hint(),
            WithDtype(param->type_annotation, tt->dtype));
          prefix_binding_map_[mid_param] = pre_param;
          // return new parameter, then we can use
          // `FreeVars` at the end of the pass to generate
          // new `mid_func` type signature
          new_args.push_back(mid_param);
        }
      }
      return relay::Call(call->op, new_args, call->attrs);
    }
  }

  LetList& prefix_ll() {
    return prefix_ll_;
  }

  std::unordered_map<relay::Var, relay::Var, ObjectPtrHash, ObjectPtrEqual>& prefix_binding_map() {
    return prefix_binding_map_;
  }

 private:
  std::set<relay::Var> params_;
  std::set<DLDataType> quantized_dtypes_;
  std::set<relay::Var> subtree_params_;
  LetList prefix_ll_;
  std::unordered_map<relay::Var, relay::Var, ObjectPtrHash, ObjectPtrEqual> prefix_binding_map_;
};

// Extract input quantization expressions from `mod['main']`.
//
// Parameters
// ----------
// mod : tvm.IRModule
//     Module containing a quantized inference function
//
// quantized_dtypes : Set[str]
//     Set of data types allowed in quantized operators
//
// Returns
// -------
// pre_mod : tvm.IRModule
//     Module containing the input quantization function
//
// mid_mod : tvm.IRModule
//     Module containing a function with everything except for input quantization

// we return *modules* so we get type checking (important for subsequent
// phases).
// TODO maybe just run type infer directly?
std::tuple<IRModule, IRModule> PartitionPrefix(const IRModule& mod, std::set<DLDataType> quantized_dtypes) {
  CHECK_EQ(mod->functions.size(), 1);
  auto func = GetRef<Function>(mod->Lookup("main").as<FunctionNode>());
  PrefixCutter prefix_cutter(func->params, quantized_dtypes);
  const Expr& mid_body = prefix_cutter.VisitExpr(func->body);
  CHECK_EQ(func->type_params.size(), 0) << "unimplemented";
  CHECK_EQ(func->attrs->dict.size(), 0) << "unimplemented";
  const relay::Function mid_func = relay::Function(
    relay::FreeVars(mid_body),
    mid_body,
    relay::Type(),
    {});
  const IRModule& mid_mod = IRModule::FromExpr(mid_func);
  LetList& prefix_ll = prefix_cutter.prefix_ll();
  // make sure we pass through all inputs in the prefix function's return expr
  // (even those that don't require quantization)
  Array<relay::Expr> ret_tuple_fields;
  auto prefix_bind_map = prefix_cutter.prefix_binding_map();
  for (const auto param : mid_func->params) {
    const auto it = prefix_bind_map.find(param);
    if (it != prefix_bind_map.end()) {
      // this param required a conversion, so we collected it in the
      // prefix cutter pass, and we can use the pass's mapping from mid
      // func params to pre func params
      ret_tuple_fields.push_back(prefix_bind_map[param]);
    } else {
      // there was no detected conversion for this argument, so we thread
      // it through the prefix function untouched
      ret_tuple_fields.push_back(relay::Var(param->name_hint(), param->checked_type()));
    }
  }
  const Expr& ret_expr = relay::Tuple(ret_tuple_fields);
  const Expr& pre_func_body = prefix_ll.Get(ret_expr);
  const Expr& pre_func = relay::Function(
    relay::FreeVars(pre_func_body),
    pre_func_body,
    relay::Type(),
    {});
  const IRModule& pre_mod = IRModule::FromExpr(pre_func);
  return std::make_tuple(pre_mod, mid_mod);
}

IRModule PartitionConversions(const IRModule& mod, std::set<DLDataType> quantized_dtypes) {
}

// TVM_REGISTER_GLOBAL("relay._quantize.PartitionConversions")
//     .set_body_typed(PartitionConversions);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
