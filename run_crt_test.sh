#!/usr/bin/env bash
python3 -u tests/python/unittest/test_crt.py 2>&1 | tee ../log.txt
