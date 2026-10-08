"""Opt-in fast projectors for the LangGraph-MAR pipeline (the original code paths are unchanged unless you call this).

    from utils.fastproj import install_fastproj
    install_fastproj()          # before building the graph: rebinds fp/bp in utils.projection, utils.algorithm, utils.graph

Requires the `recon_fast` extension (CT_recon_fanbeam_python_openmp/recon_fast, `python setup_fast.py build_ext --inplace`,
then copy recon_fast*.so next to recon*.so in code/).

Equivalence to the original `recon` extension (checked by scripts/verify_fastproj.py):
  * FP is bit-identical.
  * BP is bit-identical to one member of the original's run-to-run result set. The original BP is a static-schedule OpenMP
    reduction whose copies are added in arrival order; `P` must equal the original's team size to reproduce its partition
    (libgomp uses all visible cores when OMP_NUM_THREADS is unset, 128 on the box the paper experiments ran on). With P fixed
    the result does not depend on how many threads run it, so you may set OMP_NUM_THREADS freely *with this module*.
    Setting OMP_NUM_THREADS with the original extension changes its partition and therefore its result.
"""
import numpy as np


def install_fastproj(P=128):
    import recon_fast as RF
    import utils.projection as UP
    import utils.algorithm as UA
    import utils.graph as UG

    g = lambda p: (p["deg"], p["nview"], p["DSD"], p["DSO"], p["nx"], p["ny"], p["dx"], p["dy"], p["nu"], p["da"], p["off_a"])
    # same dtypes/shapes as the original wrappers: float64 arrays holding float32 values
    fp_ = lambda img, p: RF.FPd(img, *g(p), 3).astype(np.float64).reshape(p["nview"], p["nu"])
    bp_ = lambda proj, p: RF.BPd(proj, *g(p), P, 5).astype(np.float64).reshape(p["nx"], p["ny"])
    for m in (UP, UA, UG):
        m.fp, m.bp = fp_, bp_
    return RF.__file__
