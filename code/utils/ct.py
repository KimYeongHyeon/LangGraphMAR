import os
import platform
import uuid
from ctypes import c_int, c_float

import gecatsim as xc
import numpy as np
from gecatsim.pyfiles.CommonTools import (
    CFG, load_C_lib, rawread, rawwrite, feval,
    loadImageVolume, saveSingleImages, displayImagePictures, saveImagePictureFiles,
)
from gecatsim.reconstruction.pyfiles import recon
from numpy.ctypeslib import ndpointer


def DD2FanProj(nrdet, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr):
    clib = load_C_lib()

    func = clib.DD2FanProj
    func.argtypes = [c_int, c_float, c_float, ndpointer(c_float), ndpointer(c_float), c_float, c_float, ndpointer(c_float), c_int, ndpointer(c_float), c_int, c_int, ndpointer(c_float)]
    func.restype = None

    func(nrdet, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr)

    return sinogram

#=======================================
# setting vendor neutral cfgs
#=======================================
def AAPMRecon_init(inp_file, FOV, kernelType='R-L', unit='HU', saveImageVolume=True):
    assert kernelType in ['R-L', 'S-L', 'soft', 'standard', 'bone', 'none']
    assert unit in ['HU', '/mm', '/cm']
    cfg = CFG()

    # Phantom
    cfg.phantom.callback = "Phantom_Voxelized"      # name of function that reads and models phantom
    cfg.phantom.projectorCallback = "C_Projector_Voxelized" # name of function that performs projection through phantom
    cfg.phantom.filename = 'CatSim_logo_1024.json'  # phantom filename, not actually used in AAPM Recon
    cfg.phantom.centerOffset = [0.0, 0.0, 0.0]      # offset of phantom center relative to origin (in mm)
    cfg.phantom.scale = 1                           # re-scale the size of phantom
    if platform.system() == "Linux":
        cfg.phantom.projectorNumThreads = 4
    elif platform.system() == "Windows":
        cfg.phantom.projectorNumThreads = 1
    else:
        cfg.phantom.projectorNumThreads = 1
    
    # physics
    cfg.physics.energyCount = 12                    # number of energy bins
    cfg.physics.monochromatic = -1                  # -1 for polychromatic (see protocol.cfg);
    cfg.physics.colSampleCount = 1                  # number of samples of detector cells in lateral direction
    cfg.physics.rowSampleCount = 1                  # number of samples of detector cells in longitudinal direction
    cfg.physics.srcXSampleCount = 2                 # number of samples of focal spot in lateral direction
    cfg.physics.srcYSampleCount = 2                 # number of samples of focal spot cells in longitudinal direction
    cfg.physics.viewSampleCount = 2                 # number of samples of each view angle range in rotational direction
    cfg.physics.recalcDet = 0                       # recalculate detector geometry
    cfg.physics.recalcSrc = 0                       # recalculate source geometry and relative intensity
    cfg.physics.recalcRayAngle = 0                  # recalculate source-to-detector-cell ray angles
    cfg.physics.recalcSpec = 0                      # recalculate spectrum
    cfg.physics.recalcFilt = 0                      # recalculate filters
    cfg.physics.recalcFlux = 0                      # recalculate flux
    cfg.physics.recalcPht = 0                       # recalculate phantom
    cfg.physics.enableQuantumNoise = 1              # enable quantum noise
    cfg.physics.enableElectronicNoise = 1           # enable electronic noise
    cfg.physics.rayAngleCallback = "Detector_RayAngles_2D" # name of function to calculate source-to-detector-cell ray angles
    cfg.physics.fluxCallback = "Detection_Flux"     # name of function to calculate flux
    cfg.physics.scatterCallback = "Scatter_ConvolutionModel"                # name of function to calculate scatter
    cfg.physics.scatterKernelCallback = ""          # name of function to calculate scatter kernel ("" for default kernel)
    cfg.physics.scatterScaleFactor = 1              # scale factor, 1 appropriate for 64-mm detector and 20-cm water
    cfg.physics.callback_pre_log = "Scatter_Correction"
    cfg.physics.prefilterCallback = "Detection_prefilter" # name of function to calculate detection pre-filter
    cfg.physics.crosstalkCallback = "CalcCrossTalk" # name of function to calculate X-ray crosstalk in the detector
    cfg.physics.col_crosstalk = 0.025
    cfg.physics.row_crosstalk = 0.02
    cfg.physics.opticalCrosstalkCallback = "CalcOptCrossTalk" # name of function to calculate X-ray crosstalk in the detector
    cfg.physics.col_crosstalk_opt = 0.04
    cfg.physics.row_crosstalk_opt = 0.045
    cfg.physics.lagCallback = ""                    # name of function to calculate detector lag
    cfg.physics.opticalCrosstalkCallback = ""       # name of function to calculate optical crosstalk in the detector
    cfg.physics.DASCallback = "Detection_DAS"       # name of function to calculate the detection process
    cfg.physics.outputCallback = "WriteRawView"     # name of function to produce the simulation output
    cfg.physics.callback_post_log = 'Prep_BHC_Accurate'
    cfg.physics.EffectiveMu = 0.2
    cfg.physics.BHC_poly_order = 5
    cfg.physics.BHC_max_length_mm = 300
    cfg.physics.BHC_length_step_mm = 10
    
    # protocol
    cfg.protocol.scanTypes = [1, 1, 1]              # flags for airscan, offset scan, phantom scan
    cfg.protocol.scanTrajectory = "Gantry_Helical"  # name of the function that defines the scanning trajectory and model
    cfg.protocol.viewsPerRotation = 1000            # total numbers of view per rotation
    cfg.protocol.viewCount = 1000                   # total number of views in scan
    cfg.protocol.startViewId = 0                    # index of the first view in the scan
    cfg.protocol.stopViewId = cfg.protocol.startViewId + cfg.protocol.viewCount - 1 # index of the last view in the scan
    cfg.protocol.airViewCount = 1                   # number of views averaged for air scan
    cfg.protocol.offsetViewCount = 1                # number of views averaged for offset scan
    cfg.protocol.rotationTime = 1.0                 # gantry rotation period (in seconds)
    cfg.protocol.rotationDirection = 1              # gantry rotation direction (1=CW, -1 CCW, seen from table foot-end)
    cfg.protocol.startAngle = 0                     # relative to vertical y-axis (n degrees)
    cfg.protocol.tableSpeed = 0                     # speed of table translation along positive z-axis (in mm/sec)
    cfg.protocol.startZ = 0                         # start z-position of table
    cfg.protocol.tiltAngle = 0                      # gantry tilt angle towards negative z-axis (in degrees)
    cfg.protocol.wobbleDistance = 0.0               # focalspot wobble distance
    cfg.protocol.focalspotOffset = [0, 0, 0]        # focalspot position offset
    cfg.protocol.mA = 500                           # tube current (in mA)
    cfg.protocol.spectrumCallback = "Spectrum"      # name of function that reads and models the X-ray spectrum
    cfg.protocol.spectrumFilename = "xcist_kVp120_tar7_bin1.dat" # name of the spectrum file
    cfg.protocol.spectrumUnit_mm = 1               # Is the spectrum file in units of photons/sec/mm^2/<current>?
    cfg.protocol.spectrumUnit_mA = 1               # Is the spectrum file in units of photons/sec/<area>/mA?
    cfg.protocol.spectrumScaling = 1                # scaling factor, works for both mono- and poly-chromatic spectra
    cfg.protocol.bowtie = "large.txt"               # name of the bowtie file (or [] for no bowtie)
    cfg.protocol.filterCallback = "Xray_Filter"     # name of function to compute additional filtration
    cfg.protocol.flatFilter = ['air', 0.001]        # additional filtration - materials and thicknesses (in mm)
    cfg.protocol.dutyRatio = 1.0                    # tube ON time fraction (for pulsed tubes)
    cfg.protocol.maxPrep = -1                       # set the upper limit of prep, non-positive will disable this feature
    
    # Scanner
    cfg.scanner.detectorCallback = "Detector_ThirdgenCurved" # name of function that defines the detector shape and model
    cfg.scanner.sid = 550.0                         # source-to-iso distance (in mm)
    cfg.scanner.sdd = 950.0                         # source-to-detector distance (in mm)
    cfg.scanner.detectorColsPerMod = 1              # number of detector columns per module
    cfg.scanner.detectorRowsPerMod = 1              # number of detector rows per module
    cfg.scanner.detectorColOffset = -1.25             # detector column offset relative to centered position (in detector columns)
    cfg.scanner.detectorRowOffset = 0.0             # detector row offset relative to centered position (in detector rows)
    cfg.scanner.detectorColSize = 1.0               # detector column pitch or size (in mm)
    cfg.scanner.detectorRowSize = 1.0               # detector row pitch or size (in mm)
    cfg.scanner.detectorColCount = 900              # total number of detector columns
    cfg.scanner.detectorRowCount = cfg.scanner.detectorRowsPerMod     # total number of detector rows
    cfg.scanner.detectorPrefilter = []              # detector filter 
    cfg.scanner.focalspotCallback = "SetFocalspot"  # name of function that defines the focal spot shape and model
    cfg.scanner.focalspotData = "vct_large_fs.npz"  # Parameterize the model
    cfg.scanner.targetAngle = 7.0                   # target angle relative to scanner XY-plane (in degrees)
    cfg.scanner.focalspotWidth = 1.0
    cfg.scanner.focalspotLength = 1.0
    cfg.scanner.focalspotWidthThreshold =0.5
    cfg.scanner.focalspotLengthThreshold =0.5
    
    # Detector
    cfg.scanner.detectorMaterial = "GOS"            # detector sensor material
    cfg.scanner.detectorDepth = 3.0                 # detector sensor depth (in mm)
    cfg.scanner.detectionCallback = "Detection_EI"  # name of function that defines the detection process (conversion from X-rays to detector signal)
    cfg.scanner.detectionGain = 0.1                 # factor to convert energy to electrons (electrons / keV)
    cfg.scanner.detectorColFillFraction = 0.9       # active fraction of each detector cell in the column direction
    cfg.scanner.detectorRowFillFraction = 0.9       # active fraction of each detector cell in the row direction
    cfg.scanner.eNoise = 25                         # standard deviation of Gaussian electronic noise (in electrons)
    
    # recon
    cfg.recon.fov = FOV                           # diameter of the reconstruction field-of-view (in mm)
    cfg.recon.imageSize = 512                       # number of columns and rows to be reconstructed (square)
    cfg.recon.sliceCount = 1                        # number of slices to reconstruct
    cfg.recon.sliceThickness = 0.579                 # reconstruction slice thickness AND inter-slice interval (in mm)
    cfg.recon.centerOffset = [0.0, 0.0, 0.0]        # reconstruction offset relative to center of rotation (in mm)
    cfg.recon.reconType = 'fdk_equiAngle'           # Name of the recon function to call
    cfg.recon.kernelType = kernelType               # 'R-L' for the Ramachandran-Lakshminarayanan (R-L) filter, rectangular window function
    cfg.recon.startAngle = 0                        # in degrees; 0 is with the X-ray source at the top
    cfg.recon.unit = unit                           # '/mm', '/cm', or 'HU'
    cfg.recon.mu = 0.02                             # in /mm; typically around 0.02/mm
    cfg.recon.huOffset = -1000                      # unit is HU, -1000 HU by definition but sometimes something else is preferable
    cfg.recon.printReconParameters = False          # Flag to print the recon parameters
    cfg.recon.saveImageVolume = saveImageVolume                # Flag to save recon results as one big file
    cfg.recon.saveSingleImages = False              # Flag to save recon results as individual imagesrecon.printReconParameters = False      # Flag to print the recon parameters
    cfg.recon.displayImagePictures = False          # Flag to display the recon results as .png images
    cfg.recon.saveImagePictureFiles = False         # Flag to save the recon results as .png images
    cfg.recon.displayImagePictureAxes = False       # Flag to display the axes on the .png images
    cfg.recon.displayImagePictureTitles = False     # Flag to display the titles on the .png images

    cfg.resultsName = inp_file.split('.')[0]

    if cfg.physics.monochromatic>0:
        cfg.recon.mu = xc.GetMu('water', cfg.physics.monochromatic)[0]/10

    cfg.do_Recon = 1
    cfg.waitForKeypress = 0

    return cfg

def get_FOV(anatomy):
    if anatomy.lower() in ['head', 'h']:
        FOV = 220.16
    else:
        FOV = 400
    return FOV

class AARM():
    def __init__(self):
        pass
    def get_FOV(self, anatomy):
        if anatomy.lower() == 'h':
            FOV = 220.16
        else:
            FOV = 400
        return FOV
    def projection(self, path, anatomy):
        FOV = self.get_FOV(anatomy)
        inp_file = path

        sid = 550.
        sdd = 950.
        
        nrdetcols = 900
        nrcols = 512
        nrrows = 512
        pixsize = FOV/512
        nrviews = 1000
        
        x0 = 0.0/pixsize
        y0 = sid/pixsize
        xCor = 0.0/pixsize
        yCor = 0.0/pixsize
        
        dalpha = 2.*np.arctan2(1.0/2, sdd)
        alphas = (np.arange(nrdetcols)-(nrdetcols-1)/2-1.25)*dalpha
        xds = np.single(sdd*np.sin(alphas)/pixsize)
        yds = np.single((sid - sdd*np.cos(alphas))/pixsize)
        
        viewangles = np.single(1*(0+np.arange(nrviews)/nrviews*2*np.pi))
        if inp_file.split('.')[-1] == 'raw':
            raw_img = rawread(inp_file, [512, 512, 1], 'float')
        else:
            raise ValueError(f"Expected .raw file, got: {inp_file}")

        raw_img = raw_img/1000.*0.02+0.02 # now in the unit of mm^-1
        originalImgPtr = np.single(raw_img)
        sinogram = np.zeros([nrviews, nrdetcols, 1], dtype=np.single) 
            
        sinogram = DD2FanProj(nrdetcols, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr)
        sinogram = sinogram*pixsize
        # rawwrite(os.path.splitext(inp_file)[0]+"_DD2FanProj_900x1000.raw", sinogram)
        return sinogram
        
    def reconstruction(self, path, anatomy):
        FOV = self.get_FOV(anatomy)
        inp_file = path
        if inp_file.split('.')[-1] == 'raw':
            inp_data = rawread(inp_file, [1000, 1, 900], 'float')
            rawwrite(inp_file.replace("raw", "prep"), inp_data)
        cfg = AAPMRecon_init(inp_file, FOV)
        image = recon.recon(cfg)
        return image
    
    def _save(self):
        pass
    def _load(self):
        pass

def projection(raw, anatomy, is_metal_only=False, FOV=None):
    assert anatomy.lower() in ['body', 'head', 'b', 'h']
    if FOV is None:
        FOV = get_FOV(anatomy)
    # inp_file = path
    if isinstance(raw, str):
        if raw.split('.')[-1] == 'raw':
            raw = rawread(raw, [512, 512, 1], 'float')
        else:
            raise ValueError(f"Expected .raw file, got: {raw}")

    sid = 550.
    sdd = 950.
    
    nrdetcols = 900
    nrcols = 512
    nrrows = 512
    pixsize = FOV/512
    nrviews = 1000
    
    x0 = 0.0/pixsize
    y0 = sid/pixsize
    xCor = 0.0/pixsize
    yCor = 0.0/pixsize
    
    dalpha = 2.*np.arctan2(1.0/2, sdd)
    alphas = (np.arange(nrdetcols)-(nrdetcols-1)/2-1.25)*dalpha
    xds = np.single(sdd*np.sin(alphas)/pixsize)
    yds = np.single((sid - sdd*np.cos(alphas))/pixsize)
    
    viewangles = np.single(1*(0+np.arange(nrviews)/nrviews*2*np.pi))
    if not is_metal_only:
        raw = raw/1000.*0.02+0.02 # now in the unit of mm^-1
    originalImgPtr = np.single(raw)
    sinogram = np.zeros([nrviews, nrdetcols, 1], dtype=np.single) 
        
    sinogram = DD2FanProj(nrdetcols, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr)
    sinogram = sinogram*pixsize
    # rawwrite(os.path.splitext(inp_file)[0]+"_DD2FanProj_900x1000.raw", sinogram)
    return sinogram


def load_prep(cfg):

    print("* Loading the projection data...")
    prep = xc.rawread(cfg.resultsName + ".prep",
                  [cfg.protocol.viewCount, cfg.scanner.detectorRowCount, cfg.scanner.detectorColCount],
                  'float')
                  
    return prep

def scaleReconData(cfg, imageVolume3D):

    print('* Scaling recon data...')
    if cfg.recon.unit =='HU':
        imageVolume3D = imageVolume3D*(1000/(cfg.recon.mu)) + cfg.recon.huOffset
    elif cfg.recon.unit == '/mm':
        pass
    elif cfg.recon.unit == '/cm':
        imageVolume3D = imageVolume3D*10
    else:
        raise Exception('******** Error! An unsupported recon unit was specified: {:s}. ********'.format(cfg.recon.unit))

    return imageVolume3D

def recon(cfg):

    # If doing the recon, load the projection data, do the recon, and save the resulting image volume.
    if cfg.do_Recon:
        prep = load_prep(cfg)

        # The following line doesn't work - need to fix it when new recons are added.
        # imageVolume3D = feval("reconstruction." + cfg.recon.reconType, cfg, prep)
        imageVolume3D = feval("gecatsim.reconstruction.pyfiles." + cfg.recon.reconType, cfg, prep)

        # A hack until the previous line is fixed.
        #imageVolume3D = fdk_equiAngle(cfg, prep)
        imageVolume3D = scaleReconData(cfg, imageVolume3D)

        if cfg.recon.saveImageVolume:
            saveImageVolume(cfg, imageVolume3D)
        else:
            return imageVolume3D
    # If not doing the recon, load the previously-saved recon image volume.
    else:
        imageVolume3D = loadImageVolume(cfg)

    # In either case, save the results as individual images and display results at the specified window/level.
    if cfg.recon.saveSingleImages:
        saveSingleImages(cfg, imageVolume3D)
            
    if cfg.recon.displayImagePictures:
        cfg = displayImagePictures(cfg, imageVolume3D)

    if cfg.recon.saveImagePictureFiles:
        cfg = saveImagePictureFiles(cfg, imageVolume3D)

    return cfg

def saveImageVolume(cfg, imageVolume3D):

    print('* Writing the recon results to one big file...')

    imageVolume3D_size_string = str(cfg.recon.imageSize) + 'x' + str(cfg.recon.imageSize) + 'x' + str(cfg.recon.sliceCount)
    fname = cfg.resultsName + '_' + imageVolume3D_size_string + '.raw'
    imageVolume3D = imageVolume3D.transpose(2, 0, 1)
    imageVolume3D = imageVolume3D.copy(order='C')
    xc.rawwrite(fname, imageVolume3D)


# def reconstruction(raw, anatomy, kernelType='R-L', unit='HU', saveImageVolume=False):
#     assert anatomy.lower() in ['body', 'head', 'b', 'h']
#     # UUID를 추가하여 파일명의 고유성 보장
#     unique_id = uuid.uuid4().hex

#     if isinstance(raw, np.ndarray):
#         raw = np.ascontiguousarray(raw)
#         rawwrite(f'{unique_id}.raw', raw)
#         raw = f'{unique_id}.raw'
#     FOV = get_FOV(anatomy)
#     if isinstance(raw, str):
#         if raw.split('.')[-1] == 'raw':
#             inp_data = rawread(raw, [1000, 1, 900], 'float')
#             rawwrite(raw.replace("raw", "prep"), inp_data)
#     cfg = AAPMRecon_init(raw, FOV, kernelType=kernelType, unit=unit, saveImageVolume=saveImageVolume)
#     image = recon(cfg)
#     if isinstance(raw, str):
#         os.remove(raw)
#     return image

def reconstruction(raw_data, anatomy, kernelType='R-L', unit='HU', save_image_volume=False, FOV=None):
    """
    Reconstructs an image from raw data based on the specified anatomy and processing parameters.

    Args:
        raw_data (np.ndarray or str): The raw image data as a NumPy array or the path to the raw data file.
        anatomy (str): The part of the body for the image reconstruction, must be 'body', 'head', 'b', or 'h'.
        kernel_type (str): The type of reconstruction kernel to use. Defaults to 'R-L'.
        unit (str): The unit of measurement for the output image, typically 'HU' for Hounsfield Units. Defaults to 'HU'.
        save_image_volume (bool): Whether to save the reconstructed volume to a file. Defaults to False.

    Returns:
        np.ndarray: The reconstructed image as a NumPy array.
    """
    assert anatomy.lower() in ['body', 'head', 'b', 'h'], "Anatomy must be 'body', 'head', 'b', or 'h'."

    # Ensure the raw data has a unique filename using a UUID.
    unique_id = uuid.uuid4().hex

    if isinstance(raw_data, np.ndarray):
        # Ensure the array is contiguous in memory.
        raw_data = np.ascontiguousarray(raw_data) 
        
        raw_data = raw_data.reshape(1000, 1, 900)
        print(raw_data.shape)
        os.makedirs('tmp', exist_ok=True)
        raw_filename = f'tmp/{unique_id}.raw'
        rawwrite(raw_filename, raw_data)
        raw_data = raw_filename

    # Determine the Field of View based on anatomy.
    fov = get_FOV(anatomy)
    if FOV is not None:
        fov = FOV

    if isinstance(raw_data, str) and raw_data.endswith('.raw'):
        inp_data = rawread(raw_data, [1000, 1, 900], 'float')
        processed_filename = raw_data.replace("raw", "prep")
        rawwrite(processed_filename, inp_data)

    # Initialize the reconstruction configuration.
    config = AAPMRecon_init(raw_data, fov, kernelType=kernelType, unit=unit, saveImageVolume=save_image_volume)
    image = recon(config)

    # Clean up the raw data file if it was created during the process.
    if isinstance(raw_data, str):
        os.remove(raw_data)
        os.remove(raw_data.replace('raw', 'prep'))

    return image

def transform_image_unit_mm_to_HU(image_unit_mm, mu=0.02, huOffset=-1000):
    """
    Transforms the image volume from the unit of mm^-1 to Hounsfield Units (HU).

    Args:
        image_unit_mm (np.ndarray): The image volume in the unit of mm^-1.
        mu (float): The linear attenuation coefficient in /mm. Defaults to 0.02.
        huOffset (int): The Hounsfield Unit offset. Defaults to -1000.
    
    Returns:
        np.ndarray: The transformed image volume in Hounsfield Units (HU).
    """
    image_unit_HU = image_unit_mm * (1000 / mu) + huOffset
    return image_unit_HU


def transform_image_unit_HU_to_mm(image_unit_HU, mu=0.02, huOffset=-1000):
    """
    Transforms the image volume from Hounsfield Units (HU) back to the unit of mm^-1.

    Args:
        image_unit_HU (np.ndarray): The image volume in Hounsfield Units (HU).
        mu (float): The linear attenuation coefficient in /mm. Defaults to 0.02.
        huOffset (int): The Hounsfield Unit offset. Defaults to -1000.
    
    Returns:
        np.ndarray: The transformed image volume back to the unit of mm^-1.
    """
    image_unit_mm = (image_unit_HU - huOffset) * (mu / 1000)
    return image_unit_mm
