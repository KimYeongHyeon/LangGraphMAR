import odl
import numpy as np


class initialization:
    """CT geometry parameters for CTMAR data.

    All units are in the same coordinate system (pixel-based, matching ODL convention).
    The key is that reco_space, src_radius, det_radius, and detector extent
    must all use consistent units.

    Scanner specs (from code/utils/ct.py):
        - sid: 550.0 mm, sdd: 950.0 mm
        - 900 detectors, 1000 views, 1.0 mm detector pitch
        - FOV body: 400 mm, FOV head: 220.16 mm
        - Image: 512x512
    """
    def __init__(self, anatomy='body'):
        self.param = {}

        if anatomy.lower() in ['head', 'h']:
            fov = 220.16
        else:
            fov = 400.0

        nx_h = 512
        ny_h = 512
        self.reso = fov / nx_h  # mm per pixel

        # Image domain (in mm — ODL reco_space uses physical units)
        self.param['nx_h'] = nx_h
        self.param['ny_h'] = ny_h
        self.param['sx'] = nx_h * self.reso  # = fov
        self.param['sy'] = ny_h * self.reso  # = fov

        # View angles
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi
        self.param['nProj'] = 1000

        # Detector
        self.param['nu_h'] = 900

        # Fan beam distances (in mm — same unit as reco_space)
        self.param['dso'] = 550.0   # source-to-origin (mm)
        self.param['dde'] = 400.0   # origin-to-detector (mm) = sdd - sid = 950 - 550

        # Detector extent (in mm)
        det_col_size = 1.0  # mm per detector pixel
        self.param['su'] = det_col_size * self.param['nu_h']  # 900 mm total

        self.param['u_water'] = 0.192


def build_gemotry(param):
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0],
        shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')

    angle_partition = odl.uniform_partition(
        param.param['startangle'],
        param.param['endangle'],
        param.param['nProj'])

    detector_partition_h = odl.uniform_partition(
        -(param.param['su'] / 2.0),
        (param.param['su'] / 2.0),
        param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(
        angle_partition,
        detector_partition_h,
        src_radius=param.param['dso'],
        det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
    return ray_trafo_hh
