"""
Compatibility shim — the real script was moved to tools/debugging/check_classes_runtime.py
Importing the module will run the same behavior so old commands still work.
"""
from tools.debugging import check_classes_runtime  # noqa: F401
