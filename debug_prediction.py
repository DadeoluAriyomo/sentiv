"""
Compatibility shim — the real script was moved to tools/debugging/debug_prediction.py
Importing the module will run the same behavior so old commands still work.
"""
from tools.debugging import debug_prediction  # noqa: F401
