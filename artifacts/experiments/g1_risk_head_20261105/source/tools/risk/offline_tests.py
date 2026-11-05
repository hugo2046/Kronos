"""Explicit reviewed synthetic tests, with inherited kernel-level network denial."""
import ctypes
import errno
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.update(DOLPHINDB_URI='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
# seccomp is inherited across fork/exec and also covers native DDB clients.
lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
lib.seccomp_init.argtypes = [ctypes.c_uint32]; lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]; lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x7fff0000)
for name in (b'connect', b'sendto', b'sendmsg', b'sendmmsg'):
    if lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, lib.seccomp_syscall_resolve_name(name), 0):
        raise RuntimeError('cannot install network denial')
if lib.seccomp_load(ctx): raise RuntimeError('cannot activate network denial')
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
import qlib

def forbidden(*a, **k):
    raise RuntimeError('offline tests: database initialization forbidden')
qlib.auto_init = forbidden
qlib.init = forbidden
import pytest
# Both files inspected in full. No legacy integration file is collected.
FILES = ["tests/test_g1_risk_data.py", "tests/test_g1_risk_model.py", "tests/test_g1_risk_training.py", "tests/test_g1_risk_score.py"]
chosen = sys.argv[1:] or FILES
if any(f not in FILES for f in chosen): raise SystemExit("unreviewed test selection")
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp=/tmp/g1-risk-tests", *chosen]))
