"""Make a conda env usable without activation: put its DLL directories on PATH.

Without this, numpy's BLAS fails with a delay-load error (0xc06d007f) when the env's python.exe is
called directly. Import this module before numpy.
"""

import os
import sys

if os.name == "nt":
    _dirs = [os.path.join(sys.prefix, "Library", "bin"), sys.prefix]
    os.environ["PATH"] = os.pathsep.join(_dirs + [os.environ.get("PATH", "")])
