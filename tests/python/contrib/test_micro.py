import tvm
import os
import logging
import time

import numpy as np
from tvm.contrib import util

def test_add():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)

    def check_c():
        f1 = tvm.lower(s, [A, B, C], name="fadd")
        fsplits = [x for x in tvm.ir_pass.SplitHostDevice(f1)]
        fsplits[0] = tvm.ir_pass.LowerTVMBuiltin(fsplits[0])
        mhost = tvm.codegen.build_module(fsplits[0], "c")
        temp = util.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        #m = tvm.module.load(path_dso, "micro_dev")
        m = tvm.module.load("test.obj", "micro_dev")
        #fadd = m['fadd']
        #ctx = tvm.micro_dev(dev_id=0)
        ## launch the kernel.
        #n = nn
        #a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        #b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        #c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        #print(a)
        #print(b)
        #print(c)
        #fadd(a, b, c)
        ##print(c)
        #tvm.testing.assert_allclose(
        #    c.asnumpy(), a.asnumpy() + b.asnumpy())
    check_c()

def test_micro_array():
    pass

if __name__ == "__main__":
    test_add()
    test_micro_array()
