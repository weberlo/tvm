# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#pylint: disable=unused-argument, not-context-manager
"""Utilities for partitioning input quantization and output dequantization subexpressions."""
import numpy as np
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.type_functor import TypeMutator, TypeVisitor
from tvm.relay import transform

# operators that are allowed to defy any `quantized_dtypes` restrictions, because
# they are ops that are used in the conversion to and from the allowed
# datatypes
ALLOWED_CONVERSION_OPS = ['add', 'multiply', 'right_shift', 'clip', 'round', 'cast']

def with_dtype(ty, target_dtype):
    class DtypeReplacer(TypeMutator):
        def __init__(self, target_dtype):
            self.target_dtype = target_dtype

        def visit_tensor_type(self, tt):
            return relay.TensorType(tt.shape, self.target_dtype)

    return DtypeReplacer(target_dtype).visit(ty)


class PrefixCutter(ExprMutator):
    def __init__(self, params, quantized_dtypes):
        ExprMutator.__init__(self)
        self.params = set(params)
        self.quantized_dtypes = quantized_dtypes
        self.subtree_params = set()
        self.new_func_params = []
        self.prefix_sb = relay.ScopeBuilder()
        self.prefix_binding_map = {}

    def visit_var(self, var):
        if var in self.params:
            self.subtree_params.add(var)
        return var

    def visit_call(self, call):
        # TODO use graph pattern matching?
        if not hasattr(call.op, 'name') or call.op.name not in ALLOWED_CONVERSION_OPS:
            new_args = []
            for arg in call.args:
                new_arg = self.visit(arg)
                if len(self.subtree_params) == 0:
                    new_args.append(new_arg)
                else:
                    assert len(self.subtree_params) == 1
                    param = next(iter(self.subtree_params))
                    pre_param = self.prefix_sb.let(param.name_hint, new_arg)
                    self.subtree_params.clear()
                    mid_param = relay.Var(param.name_hint, with_dtype(param.type_annotation, arg.checked_type.dtype))
                    self.prefix_binding_map[mid_param] = pre_param
                    # return new parameter, then we can use
                    # relay.analysis.free_vars at the end of the pass to generate
                    # new `mid_func` type signature
                    new_args.append(mid_param)
            return relay.Call(call.op, new_args, call.attrs)
        else:
            return super().visit_call(call)


def partition_prefix(mod, quantized_dtypes):
    assert len(mod.functions) == 1
    func = mod['main']
    prefix_cutter = PrefixCutter(func.params, quantized_dtypes)
    mid_body = prefix_cutter.visit(func.body)
    assert not func.type_params, 'unimplemented'
    assert func.attrs is None, 'unimplemented'
    mid_func = relay.Function(
        relay.analysis.free_vars(mid_body),
        mid_body)
    mid_mod = tvm.IRModule.from_expr(mid_func)

    scope_builder = prefix_cutter.prefix_sb
    # make sure we pass through all inputs in the prefix function's return expr
    # (even those that don't require quantization)
    ret_expr = []
    for param in mid_func.params:
        if param in prefix_cutter.prefix_binding_map:
            # this param required a conversion, so we collected it in the
            # prefix cutter pass, and we can use the pass's mapping from mid
            # func params to pre func params
            ret_expr.append(prefix_cutter.prefix_binding_map[param])
        else:
            # there was no detected conversion for this argument, so we thread
            # it through the prefix function untouched
            ret_expr.append(relay.Var(param.name_hint, param.checked_type))
    ret_expr = relay.Tuple(ret_expr)
    scope_builder.ret(ret_expr)
    pre_func_body = scope_builder.get()
    pre_func = relay.Function(relay.analysis.free_vars(pre_func_body), pre_func_body)
    pre_mod = tvm.IRModule.from_expr(pre_func)

    return pre_mod, mid_mod


class SuffixCutter(ExprMutator):
    def __init__(self, quantized_dtypes):
        ExprMutator.__init__(self)
        self.mid_body = None
        self.quantized_dtypes = quantized_dtypes

    def visit(self, expr):
        if hasattr(expr, 'checked_type') and expr.checked_type.dtype in self.quantized_dtypes:
            self.mid_body = expr
            return relay.Var('input', expr.checked_type)
        else:
            return super().visit(expr)


def partition_suffix(mod, quantized_dtypes):
    assert len(mod.functions) == 1
    func = mod['main']
    suffix_cutter = SuffixCutter(quantized_dtypes)
    post_body = suffix_cutter.visit(func.body)
    assert not func.type_params, 'unimplemented'
    assert func.attrs is None, 'unimplemented'
    post_func = relay.Function(
        relay.analysis.free_vars(post_body),
        post_body,
        func.ret_type)
    post_mod = tvm.IRModule.from_expr(post_func)

    mid_body = suffix_cutter.mid_body
    if mid_body is None:
        # The suffix contains the entire function, meaning there was no
        # quantization boundary in the given mod.  In this case, we use the
        # suffix mod as the middle mod and make the suffix an identity function.
        mid_mod = post_mod
        post_body = relay.Var('input', mid_mod['main'].ret_type)
        post_func = relay.Function(
            [post_body],
            post_body)
        post_mod = tvm.IRModule.from_expr(post_func)
    else:
        mid_func = relay.Function(
            func.params,
            mid_body)
        mid_mod = tvm.IRModule.from_expr(mid_func)

    return mid_mod, post_mod


def fuse_partitions(pre_mod, mid_mod, post_mod):
    pre_func = pre_mod['main']
    mid_func = mid_mod['main']
    post_func = post_mod['main']
    fused_mod = tvm.IRModule(functions={
        relay.GlobalVar('quantize_inputs'): pre_func,
        relay.GlobalVar('quantized_main'): mid_func,
        relay.GlobalVar('dequantize_outputs'): post_func,
    })

    sb = relay.ScopeBuilder()
    fused_mod_main_params = [relay.Var(param.name_hint) for param in pre_func.params]
    quantized_inputs = sb.let('quantized_inputs', relay.Call(
        fused_mod.get_global_var('quantize_inputs'),
        fused_mod_main_params
    ))
    quantized_outputs = sb.let('quantized_outputs', relay.Call(
        fused_mod.get_global_var('quantized_main'),
        [relay.TupleGetItem(quantized_inputs, i) for i in range(len(pre_func.ret_type.fields))]
    ))
    dequantized_outputs = sb.let('dequantized_outputs', relay.Call(
        fused_mod.get_global_var('dequantize_outputs'),
        [quantized_outputs]
    ))
    sb.ret(dequantized_outputs)
    fused_mod['main'] = relay.Function(fused_mod_main_params, sb.get())
    return fused_mod


class ValidStarfixChecker(ExprVisitor):
    """Checks that the given prefix or suffix (i.e., *fix) contains only conversion ops"""
    def __init__(self):
        ExprVisitor.__init__(self)
        self.valid = True

    def visit_call(self, call):
        if not hasattr(call.op, 'name') or call.op.name not in ALLOWED_CONVERSION_OPS:
            self.valid = False
        super().visit_call(call)


def has_only_conversion_ops(func):
    checker = ValidStarfixChecker()
    checker.visit(func)
    return checker.valid


def partition_conversions(mod, quantized_dtypes):
    assert len(mod.functions) == 1
    pre_mod, mid_mod = partition_prefix(mod, quantized_dtypes)
    mid_mod, post_mod = partition_suffix(mid_mod, quantized_dtypes)
    assert has_only_conversion_ops(pre_mod['main'])
    assert relay.analysis.all_dtypes(mid_mod['main']).issubset(quantized_dtypes)
    assert has_only_conversion_ops(post_mod['main'])
    return fuse_partitions(pre_mod, mid_mod, post_mod)
