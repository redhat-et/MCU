"""Pytest configuration file with Triton interpreter mode fixture."""

import os
import importlib

import pytest

import triton
import triton.language as tl

@pytest.fixture(scope='class', params=['0', '1'])
def triton_interpret(request):
    """
    Test Triton in both regular mode (TRITON_INTERPRET=0) and interpreter mode (TRITON_INTERPRET=1)

    Sets the TRITON_INTERPRET environment variable to either "0" or "1",
    reloads the Triton modules, and ensures the env var is cleaned up afterward.
    """
    os.environ['TRITON_INTERPRET'] = request.param
    importlib.reload(triton)
    importlib.reload(tl)
    yield request.param
    os.environ.pop('TRITON_INTERPRET', None)
