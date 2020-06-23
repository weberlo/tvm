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

# TODO don't hardcode so many things as int8

# TODO using "partition" could get confusing, because there's already another
# mechanism in quantization that's called partitioning (see `_partition.py`)

QUANT_DEQUANT_ALLOWED_OPS = ['add', 'multiply', 'right_shift', 'clip', 'round', 'cast']

def with_dtype(ty, target_dtype):
    class DtypeReplacer(TypeMutator):
        def __init__(self, target_dtype):
            self.target_dtype = target_dtype

        def visit_tensor_type(self, tt):
            return relay.TensorType(tt.shape, self.target_dtype)

    return DtypeReplacer(target_dtype).visit(ty)


def gen_rand_np(tt, low, high):
    if 'int' in tt.dtype:
        return np.random.randint(low, high, size=get_const_tuple(tt.shape), dtype=tt.dtype)
    elif 'float' in tt.dtype:
        return np.random.uniform(low, high, size=get_const_tuple(tt.shape)).astype(tt.dtype)
    else:
        assert False, 'unknown dtype'


def gen_rand_tvm(tt, low, high):
    return tvm.nd.array(gen_rand_np(tt, low, high), ctx=tvm.cpu(0))


class QuantPrefixCutter(ExprMutator):
    def __init__(self, params):
        ExprMutator.__init__(self)
        self.params = set(params)
        self.subtree_params = set()
        self.new_func_params = []
        self.prefix_sb = relay.ScopeBuilder()

    def visit_var(self, var):
        if var in self.params:
            self.subtree_params.add(var)
        return var

    def visit_call(self, call):
        if call.op.name == 'cast' and call.attrs['dtype'] == 'int8':
            res = super().visit_call(call)
            if len(self.subtree_params) == 0:
                return res
            else:
                assert len(self.subtree_params) == 1
                param = next(iter(self.subtree_params))
                self.prefix_sb.let(param.name_hint, res)
                self.subtree_params.clear()
                # return new parameter, then we can use
                # relay.analysis.free_vars at the end of the pass to generate
                # new `mid_func` type signature
                return relay.Var(param.name_hint, with_dtype(param.type_annotation, 'int8'))
        else:
            return super().visit_call(call)

    def visit_function(self, func):
        # override to make sure we *don't* visit the function params
        return relay.Function(
            func.params,
            self.visit(func.body),
            func.ret_type,
            func.type_params,
            func.attrs)


def partition_prefix(mod):
    assert len(mod.functions) == 1
    func = mod['main']
    prefix_cutter = QuantPrefixCutter(func.params)
    mid_body = prefix_cutter.visit(func.body)
    assert not func.type_params, 'unimplemented'
    assert func.attrs is None, 'unimplemented'
    mid_func = relay.Function(
        relay.analysis.free_vars(mid_body),
        mid_body)
    mid_mod = tvm.IRModule.from_expr(mid_func)

    scope_builder = prefix_cutter.prefix_sb
    ret_expr = relay.Tuple(list(map(lambda b: b[0], scope_builder._bindings[-1])))
    scope_builder.ret(ret_expr)
    pre_func_body = scope_builder.get()
    pre_func = relay.Function(relay.analysis.free_vars(pre_func_body), pre_func_body)
    pre_mod = tvm.IRModule.from_expr(pre_func)

    return pre_mod, mid_mod


class QuantSuffixCutter(ExprMutator):
    def __init__(self):
        ExprMutator.__init__(self)
        self.mid_body = None

    def visit(self, expr):
        if hasattr(expr, 'checked_type') and expr.checked_type.dtype == 'int8':
            self.mid_body = expr
            return relay.Var('input', expr.checked_type)
        else:
            return super().visit(expr)


class ForbiddenOpException(Exception):
    def __init__(self, forbidden_op, is_prefix):
        self.forbidden_op = forbidden_op
        self.is_prefix


class OpChecker(ExprVisitor):
    def __init__(self, allowed_ops, is_prefix):
        ExprVisitor.__init__(self)
        self.allowed_ops = allowed_ops
        self.is_prefix = is_prefix

    def visit_call(self, call):
        if call.op.name not in self.allowed_ops:
            raise ForbiddenOpException(call.op.name, self.is_prefix)


class ForbiddenDtypeException(Exception):
    def __init__(self, invalid_dtype, allowed_dtypes):
        self.invalid_dtype = invalid_dtype
        self.allowed_dtypes = allowed_dtypes


class TyDtypeChecker(TypeVisitor):
    def __init__(self, allowed_dtypes):
        TypeVisitor.__init__(self)
        self.allowed_dtypes = allowed_dtypes

    def visit_tensor_type(self, tt):
        if tt.dtype not in self.allowed_dtypes:
            raise ForbiddenDtypeException(tt.dtype, self.allowed_dtypes)


class DtypeChecker(ExprVisitor):
    def __init__(self, allowed_dtypes):
        ExprVisitor.__init__(self)
        self.allowed_dtypes = allowed_dtypes
        self.ty_checker = TyDtypeChecker(allowed_dtypes)

    def visit(self, expr):
        if hasattr(expr, 'checked_type'):
            self.ty_checker.visit(expr.checked_type)
        return super().visit(expr)


def partition_suffix(mod):
    assert len(mod.functions) == 1
    func = mod['main']
    suffix_cutter = QuantSuffixCutter()
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


def partition_quantized(mod, allowed_dtypes):
    # TODO we have the prefix/suffix conversion code chopped off, but we want
    # it to be a *partition*, so we need to save the pieces we're cutting off.
    # Also, do we want any restrictions on how much gets cut off?
    # Right now, with the CIFAR-10 CNN, it cuts off some bias add and dense
    # operations (i.e., not just conversion ops like clip, cast, and round),
    # causing type inference to fail.
    #
    # should the user receive diagnostics about the results (e.g., letting them
    # know some operators were chopped off in the prefix/suffix?)
    #
    # keep in mind that we have an implicit assumption in the prefix cutter
    # that the first operator is quantizable, which is fairly safe, since a CNN
    # usually starts with a conv, but if we have plans on generalizing the
    # quantization pass to arbitrary models, this will need to change.
    assert len(mod.functions) == 1
    pre_mod, mid_mod = partition_prefix(mod)
    if len(pre_mod['main'].params) == 0:
        mod_str = '  ' + str(mod).replace('\n', '\n  ')
        raise RuntimeError(f'no inputs to be quantized in given mod (shown below):\n{mod_str}')
    mid_mod, post_mod = partition_suffix(mid_mod)

    try:
        OpChecker(QUANT_DEQUANT_ALLOWED_OPS, True).visit(pre_mod['main'])
        DtypeChecker(allowed_dtypes).visit(mid_mod['main'])
        OpChecker(QUANT_DEQUANT_ALLOWED_OPS, False).visit(post_mod['main'])
    # TODO we should be able to unify the forbidden dtype and forbidden op
    # exceptions, since they're pointing out essentially the same thing
    # (there's an unquantizable op in your quantized result)
    except ForbiddenDtypeException as e:
        mod_str = '  ' + str(mid_mod).replace('\n', '\n  ')
        raise RuntimeError(f'found dtype `{e.invalid_dtype}` in middle partition'
            f' when allowed dtypes were `{e.allowed_dtypes}`. module shown below:\n{mod_str}')
    except ForbiddenOpException as e:
        starfix_str = 'input quantization prefix' if e.prefix else 'output dequantization suffix'
        raise RuntimeError(f'found op `{e.forbidden_op}` in {starfix_str}')

    return pre_mod, mid_mod, post_mod
