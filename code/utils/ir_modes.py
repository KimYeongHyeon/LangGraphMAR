"""Opt-in handling of the iterative-reconstruction (IR) node's output `img_m` (the metal component).

The workflow in utils/graph.py runs IR on every trial of the threshold search and keeps `img_m` only for the last trial,
while the search itself (thresholding, inpainting, enhancement, ground checking) never reads it. To build the full image
(metal re-inserted) from the *best* trial you need that trial's `img_m`:

  RecordIR    keeps the graph exactly as published and copies each trial's `img_m` (default; no behavioural change).
  DeferredIR  skips IR on every trial and runs it once for the best trial from a captured `sino_m = original_sinogram - sino_b`.
              This changes the *execution* of the workflow (not the nodes, their order, the search or the decisions) and is
              markedly faster, because IR dominates the per-trial cost. Results matched RecordIR within the run-to-run noise of
              the original on the checked slices (see README, History).

Both must be installed before `get_workflow_graph()` is called, because the graph reads the node functions from `utils.graph`.
Pixel size: `utils.algorithm.IterativeReconstruction` builds its image grid from a fixed field of view (400 mm for body,
220.16 mm for head: constants of the AAPM dataset). For data with another pixel size the metal image would come out at the wrong
scale and, added to the final image, produce large errors. Set `SLICE_FOV["mm"] = pixel_size_mm * 512` for every case and call
`install_slice_fov(UG)` once; both modes then reconstruct on the same geometry as `fp`/`bp` of that case (the class already
accepts a `FOV` argument; nothing else changes, and the AAPM data is unaffected when you leave `SLICE_FOV["mm"]` as None).

Look up a trial by its image-mask threshold (it decreases by 0.01 per trial, so it is unique within a case):

    import utils.graph as UG
    ir = RecordIR()                      # or DeferredIR()
    ir.install(UG)
    app = UG.get_workflow_graph(mode="experiment")
    ...                                  # run the graph for one case; call ir.att.clear() between cases
    img_m = ir.img_m(best_params["image_mask_threshold"])      # mu (1/mm) of the best trial
"""


SLICE_FOV = {"mm": None}


def install_slice_fov(UG):
    """Give the graph's `IterativeReconstruction(anatomy)` call the current case's FOV (see module docstring)."""
    Orig = UG.IterativeReconstruction
    UG.IterativeReconstruction = lambda anatomy, FOV=None: Orig(anatomy, FOV if FOV is not None else SLICE_FOV["mm"])


class RecordIR:
    def __init__(self):
        self.att = []

    def install(self, UG):
        ir = UG.iterative_reconstruction

        def rec(state):
            thr = float(state["reconstruction_params"]["image_mask_threshold"])     # read before the node mutates the params
            r = ir(state)
            self.att.append({"thr": thr, "img_m": r["img_m"].copy()})
            return r

        UG.iterative_reconstruction = rec

    def img_m(self, thr):
        hit = [a for a in self.att if round(a["thr"], 2) == round(thr, 2)]
        assert len(hit) == 1, f"{len(hit)} trials with threshold {thr}"
        return hit[0]["img_m"]


class DeferredIR:
    def __init__(self):
        self.att = []

    def install(self, UG):
        def skip_ir(state):
            rp = state["reconstruction_params"]
            # The input must be captured now: utils.projection.filtering modifies its input in place, so get_data re-weights
            # original_sinogram on every trial and re-running inpainting afterwards would see a different sinogram.
            self.att.append({"thr": float(rp["image_mask_threshold"]), "sino_m": state["original_sinogram"] - state["sino_b"],
                             "anatomy": state["anatomy"], "fov": SLICE_FOV["mm"], "n_it": rp["ir_num_iterations"], "soft": rp["soft_thresholding"]})
            rp["current_IR_iteration"] += 1
            return {"reconstruction_params": rp}

        UG.iterative_reconstruction = skip_ir
        UG.generate_result = lambda state: {"img_b": state["img_b"]}     # its output (metalart_sinogram) is never read again

    def img_m(self, thr):
        from utils.algorithm import IterativeReconstruction
        hit = [a for a in self.att if round(a["thr"], 2) == round(thr, 2)]
        assert len(hit) == 1, f"{len(hit)} trials with threshold {thr}"
        a = hit[0]
        return IterativeReconstruction(a["anatomy"], a["fov"]).perform_iterative_reconstruction(a["sino_m"].copy(), num_iterations=a["n_it"], soft_thresholding=a["soft"])
