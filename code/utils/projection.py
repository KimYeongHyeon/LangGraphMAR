import os
import numpy as np
from scipy import fft
import importlib
import recon
importlib.reload(recon)

param = {}
param['nx'] = 512
param['ny'] = 512
param['dx'] = 1.0
param['dy'] = 1.0
param['DSD'] = 950.0
param['DSO'] = 550.0
param['nu'] = 900
param['du'] = 1.0
param['nview'] = 1000
param['deg'] = np.linspace(0, 360, param['nview'], endpoint=False)
param['fan_angle'] = param['du']/param['DSD']*180/3.141592*param['nu']
param['da'] = param['fan_angle']/param['nu']/180*3.141592
param['off_a'] = 0


def filtering(proj, param):
    proj = np.moveaxis(proj,0,-1)
    as_ = (-(np.arange(-param['nu']/2+0.5, param['nu']/2, 1)) - param['off_a']) * param['da']
    weight = (as_[:, np.newaxis] / 180 * np.pi / np.sin(as_[:, np.newaxis] / 180 * np.pi)) ** 2
    weight[np.isnan(weight)] = 1

    for i in range(param['nview']):
        proj[:, i] = proj[:, i] * np.cos(as_ * np.pi / 180) * weight.flatten()    
        
    filt_len = max(64, 2**int(np.ceil(np.log2(2*param['nu']))))
    ramp_kernel = ramp(filt_len)
    filt = Filter(param['filter'], ramp_kernel, filt_len)
    filt = np.tile(filt, (param['nview'], 1)).T

    fproj = np.zeros((filt_len, param['nview']), dtype=np.float32)
    fproj[filt_len//2-param['nu']//2:filt_len//2+param['nu']//2, :] = proj
    fproj = fft.fft(fproj, axis=0)
    fproj = fproj * filt
    fproj = np.real(fft.ifft(fproj, axis=0))
    proj = fproj[filt_len//2-param['nu']//2:filt_len//2+param['nu']//2, :] * (2*np.pi/param['nview']) / (param['da']*param['DSD']) / 2 / (param['dy']**2) * param['du']

    return np.moveaxis(proj,0,-1)

def ramp(n):
    nn = np.arange(-n//2, n//2)
    h = np.zeros(nn.shape, dtype=np.float32)
    h[n//2] = 1 / 4
    odd = nn % 2 == 1
    h[odd] = -1 / (np.pi * nn[odd])**2
    return h

def Filter(filter_type, kernel, order):
    f_kernel = np.abs(fft.fft(kernel))
    filt = f_kernel[:order//2+1]
    w = 2 * np.pi * np.arange(filt.size) / order  # frequency axis up to Nyquist
    
    if filter_type.lower() == 'ram-lak':
        pass
    elif filter_type.lower() == 'shepp-logan':
        filt[1:] *= np.sin(w[1:] / (2)) / (w[1:] / (2))
    elif filter_type.lower() == 'cosine':
        filt[1:] *= np.cos(w[1:] / (2))
    elif filter_type.lower() == 'hamming':
        filt[1:] *= 0.54 + 0.46 * np.cos(w[1:] )
    elif filter_type.lower() == 'hann':
        filt[1:] *= (1 + np.cos(w[1:] )) / 2
    else:
        raise ValueError(f'Invalid filter selected: {filter_type}')
    
    filt = np.concatenate([filt, filt[-2:0:-1]])  # Symmetry of the filter
    return filt


def bp(proj, param):
    img = recon.BP(proj.flatten().tolist(), param['deg'].tolist(), param['nview'], param['DSD'], param['DSO'], param['nx'], param['ny'], param['dx'], param['dy'], param['nu'], param['da'], param['off_a'])
    img =  np.reshape(img, (param['nx'], param['ny']))
    return img

def fp(img, param):
    proj = recon.FP(img.flatten().tolist(), param['deg'].tolist(), param['nview'], param['DSD'], param['DSO'], param['nx'], param['ny'], param['dx'], param['dy'], param['nu'], param['da'], param['off_a'])
    proj =  np.reshape(proj, (param['nview'],param['nu']))
    return proj