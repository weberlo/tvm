import argparse

import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import onnx
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.type_functor import TypeMutator, TypeVisitor
from tvm.relay import transform
from topi import get_const_tuple

from micro_eval import util
from micro_eval.util import model_util

# operators that are allowed to defy any `allowed_dtypes` restrictions, because
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
    def __init__(self, params, allowed_dtypes):
        ExprMutator.__init__(self)
        self.params = set(params)
        self.allowed_dtypes = allowed_dtypes
        self.subtree_params = set()
        self.new_func_params = []
        self.prefix_sb = relay.ScopeBuilder()
        self.prefix_binding_map = {}

    def visit_var(self, var):
        if var in self.params:
            self.subtree_params.add(var)
        return var

    def visit_call(self, call):
        if call.op.name == 'cast' \
                and call.args[0].checked_type.dtype not in self.allowed_dtypes \
                and call.attrs['dtype'] in self.allowed_dtypes:
            cast_dtype = call.attrs['dtype']
            res = super().visit_call(call)
            if len(self.subtree_params) == 0:
                return res
            else:
                assert len(self.subtree_params) == 1
                param = next(iter(self.subtree_params))
                pre_param = self.prefix_sb.let(param.name_hint, res)
                self.subtree_params.clear()
                mid_param = relay.Var(param.name_hint, with_dtype(param.type_annotation, cast_dtype))
                self.prefix_binding_map[mid_param] = pre_param
                # return new parameter, then we can use
                # relay.analysis.free_vars at the end of the pass to generate
                # new `mid_func` type signature
                return mid_param
        else:
            return super().visit_call(call)


def partition_prefix(mod, allowed_dtypes):
    assert len(mod.functions) == 1
    func = mod['main']
    prefix_cutter = PrefixCutter(func.params, allowed_dtypes)
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
    def __init__(self, allowed_dtypes):
        ExprMutator.__init__(self)
        self.mid_body = None
        self.allowed_dtypes = allowed_dtypes

    def visit(self, expr):
        if hasattr(expr, 'checked_type') and expr.checked_type.dtype in self.allowed_dtypes:
            self.mid_body = expr
            return relay.Var('input', expr.checked_type)
        else:
            return super().visit(expr)


def partition_suffix(mod, allowed_dtypes):
    assert len(mod.functions) == 1
    func = mod['main']
    suffix_cutter = SuffixCutter(allowed_dtypes)
    post_body = suffix_cutter.visit(func.body)
    assert not func.type_params, 'unimplemented'
    assert func.attrs is None, 'unimplemented'
    post_func = relay.Function(
        relay.analysis.free_vars(post_body),
        post_body,
        func.ret_type)
    post_mod = tvm.IRModule.from_expr(post_func)

    mid_body = suffix_cutter.mid_body
    mid_func = relay.Function(
        func.params,
        mid_body)
    mid_mod = tvm.IRModule.from_expr(mid_func)

    return mid_mod, post_mod


class UnquantizedOpCollector(ExprVisitor):
    def __init__(self, allowed_dtypes):
        ExprVisitor.__init__(self)
        self.allowed_dtypes = allowed_dtypes
        self.unquantized_ops = set()

    def visit_call(self, call):
        if hasattr(call, 'checked_type') \
                and call.checked_type.dtype not in self.allowed_dtypes \
                and call.op.name not in ALLOWED_CONVERSION_OPS:
            self.unquantized_ops.add(call.op)
        super().visit_call(call)


def collect_unquantized_ops(quantized_mod, allowed_dtypes):
    op_collector = UnquantizedOpCollector(allowed_dtypes)
    op_collector.visit(quantized_mod['main'])
    return op_collector.unquantized_ops


def partition_quantized(mod, allowed_dtypes):
    assert len(mod.functions) == 1
    pre_mod, mid_mod = partition_prefix(mod, allowed_dtypes)
    mid_mod, post_mod = partition_suffix(mid_mod, allowed_dtypes)
    return pre_mod, mid_mod, post_mod
