import tvm
import os
import logging
import time

import numpy as np
from tvm.contrib import util

def test_add():
    size = 1024
    n = tvm.convert(size)
    A_dummy = tvm.placeholder((n,), name='A')
    B_dummy = tvm.placeholder((n,), name='B')
    C_dummy = tvm.compute(A_dummy.shape, lambda *i: A_dummy(*i) + B_dummy(*i), name='C')

    m = tvm.module.load("fadd.obj", "micro_dev")
    fadd = m['fadd']
    ctx = tvm.micro_dev(dev_id=0)

    A = tvm.nd.array(np.random.uniform(size=size).astype(A_dummy.dtype), ctx)
    B = tvm.nd.array(np.random.uniform(size=size).astype(B_dummy.dtype), ctx)
    C = tvm.nd.array(np.zeros(size, dtype=C_dummy.dtype), ctx)

    print(f'A = {A}')
    print(f'B = {B}')
    print(f'C = {C}')
    print()
    input('[enter to continue]')
    print()
    print('Computing `C = A + B`...', end='')
    fadd(A, B, C)
    print('done')
    print()
    input('[enter to continue]')
    print()
    print(f'A = {A}')
    print(f'B = {B}')
    print(f'C = {C}')
    print()
    input('[enter to continue]')
    print()

    print('Asserting `C = A + B`...', end='')
    tvm.testing.assert_allclose(
        C.asnumpy(), A.asnumpy() + B.asnumpy())
    print('done')
    print()
    print('Test Passed')


def test_micro_array():
    pass

if __name__ == "__main__":
    test_add()
    test_micro_array()
