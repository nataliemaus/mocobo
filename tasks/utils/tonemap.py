
from scipy.sparse import spdiags, linalg

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pyiqa
import torch

NEW_COLOR_MAPPING_IMG_TASKS = True 

def wls_filter(log_luma, lambda_=1, alpha=1.2, rtol=1e-4):
    """
    edge-preserving smoothing via weighted least squares (WLS)
        u = F_λ (g) = (I + λ L_g)^(-1) g
        L_g = D_x^T A_x D_x +D_y^T A_y D_y

    arguments:
        luma (2-dim array, required) - the input image luma
        lambda_ (float) - balance between the data term and
            the smoothness term
        alpha (float) - a degree of control over the affinities
            by non-lineary scaling the gradients

    return:
        out (2-dim array)

    Taken from: https://github.com/zhaoyin214/wls_filter
    """
    EPS = 1e-4
    DIM_X = 1
    DIM_Y = 0

    height, width = log_luma.shape[0 : 2]
    size = height * width

    # affinities between adjacent pixels based on gradients of luma
    # dy
    diff_log_luma_y = np.diff(a=log_luma, n=1, axis=DIM_Y)
    diff_log_luma_y = - lambda_ / (np.abs(diff_log_luma_y) ** alpha + EPS)
    diff_log_luma_y = np.pad(
        array=diff_log_luma_y, pad_width=((0, 1), (0, 0)),
        mode="constant"
    )
    diff_log_luma_y = diff_log_luma_y.ravel()

    # dx
    diff_log_luma_x = np.diff(a=log_luma, n=1, axis=DIM_X)
    diff_log_luma_x = - lambda_ / (np.abs(diff_log_luma_x) ** alpha + EPS)
    diff_log_luma_x = np.pad(
        array=diff_log_luma_x, pad_width=((0, 0), (0, 1)),
        mode="constant"
    )
    diff_log_luma_x = diff_log_luma_x.ravel()

    # construct a five-point spatially inhomogeneous Laplacian matrix
    diff_log_luma = np.vstack((diff_log_luma_y, diff_log_luma_x))
    smooth_weights = spdiags(data=diff_log_luma, diags=[-width, -1],
                             m=size, n=size)

    w = np.pad(array=diff_log_luma_y, pad_width=(width, 0), mode="constant")
    w = w[: -width]
    n = np.pad(array=diff_log_luma_x, pad_width=(1, 0), mode="constant")
    n = n[: -1]

    diag_data = 1 - (diff_log_luma_x + w + diff_log_luma_y + n)
    smooth_weights = smooth_weights + smooth_weights.transpose() + \
        spdiags(data=diag_data, diags=0, m=size, n=size)

    out, _ = linalg.cg(A=smooth_weights, b=log_luma.ravel(), rtol=rtol)
    out = out.reshape((height, width))
    return out
 
def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(1 / float(tokens[1]))

    return images, np.asarray(times, dtype=np.float32)

def tonemap(hdr, lum, n_layers, gf_radiuses, gf_epsilons, detail_gains, base_gain, color_gain):
    """
    arguments:
        hdr: (3-dim array) high dynamic range image (normalized to [0, 1]).
        lum: (2-dim array) luminance image of hdr.
        n_layers: (int) number of decomposed detail layers.
        gf_radiuses: (1-dim array) `radius` parameter for the guided filter for generating the detail layers.
        gf_epsilons: (1-dim array) `epsilon` parameter for the guided filter for generating the detail layers.
        detail_gains: (1-dim array) The gain of each detail layer for tone mapping
        base_gains: (float) The gain of the base layer. Should be less than 1 for tonemapping.
        color_gain: (float) Gain of the colors. Should be close to 1 for tonemapping.
    """
    assert len(gf_radiuses)  == n_layers
    assert len(gf_epsilons)  == n_layers
    assert len(detail_gains) == n_layers

    base    = np.log(lum + 1e-10)
    details = []
    for i in range(n_layers):
        base_prev = base.copy()
        base      = cv.ximgproc.guidedFilter(base_prev, base_prev, gf_radiuses[i], gf_epsilons[i]) 
        detail    = base_prev - base
        details.append(detail)

    loglum_mapped = base_gain*base
    for i in range(n_layers):
        loglum_mapped += detail_gains[i]*details[i]

    ldr_col = np.log(hdr / np.expand_dims(lum, 2) + 1e-10)
    ldr     = np.exp(color_gain * ldr_col + np.expand_dims(loglum_mapped, 2))
    return ldr

def gamma_correction(img, gain, gamma):
    return np.pow(gain*(img - np.min(img)), gamma).clip(0, 1)

def lerp(a, b, t):
    """Linear interpolate on the scale given by a to b, using t as the point on that scale.
    adapted from https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
    """
    return (1 - t) * a + t * b

def map_parameters(params, n_detail_layers):
    idx = 0

    # Guided filter radius parameters in [3, 32]
    param_len   = n_detail_layers
    gf_radiuses = np.round(lerp(3, 32, params[idx:idx+param_len])).astype(int)
    idx        += param_len

    # Guided filter epsilon parameters in [0.01, 1.0]
    param_len    = n_detail_layers
    gf_epsilons  = lerp(0.01, 1.0, params[idx:idx+param_len])
    idx         += param_len

    # Tone mapping detail gains in [0, 1.5]
    param_len    = n_detail_layers
    detail_gains = lerp(0.0, 1.5, params[idx:idx+param_len])
    idx         += param_len

    # Tone mapping base gain in [0, 1.0]
    param_len  = 1
    base_gain  = lerp(0., 1., params[idx:idx+param_len][0])
    idx       += param_len


    if NEW_COLOR_MAPPING_IMG_TASKS: # new color mapping, try to make look nicer 
        # Tone mapping color gain in [0.5, 1.5]
        param_len  = 1
        color_gain = lerp(0.5, 1.5, params[idx:idx+param_len][0])
        idx       += param_len
    else:
        # old color mapping, images look meh 
        # Tone mapping color gain in [0, 1.0]
        param_len  = 1
        color_gain = lerp(0., 1., params[idx:idx+param_len][0])
        idx       += param_len

    
    # Gamma correction gamma parameter in [1.0, 5.0]
    param_len  = 1
    invgamma   = lerp(1., 5., params[idx:idx+param_len][0])
    idx       += param_len

    # Gamma correction gain in [0.2, 2.0]
    param_len = 1
    gain      = lerp(0.2, 2.0, params[idx:idx+param_len][0])
    idx      += param_len

    return {"gf_radiuses"  : gf_radiuses,
            "gf_epsilons"  : gf_epsilons,
            "detail_gains" : detail_gains,
            "base_gain"    : base_gain,
            "color_gain"   : color_gain,
            "invgamma"     : invgamma,
            "gain"         : gain,}

def pipeline(hdr, n_detail_layers, params_mapped):
    hdr_b = hdr[:,:,0]
    hdr_g = hdr[:,:,1]
    hdr_r = hdr[:,:,2]
    lum   = 0.2989*hdr_r + 0.587*hdr_g + 0.114*hdr_b

    ldr = tonemap(hdr, lum,
                  n_detail_layers,
                  params_mapped["gf_radiuses"],
                  params_mapped["gf_epsilons"],
                  params_mapped["detail_gains"],
                  params_mapped["base_gain"],
                  params_mapped["color_gain"])
    out = gamma_correction(ldr, params_mapped["gain"], 1/params_mapped["invgamma"])
    return out

def main():
    # Setup
    images, times = loadExposureSeq("hdr")
 
    calibrate = cv.createCalibrateDebevec()
    response = calibrate.process(images, times)
 
    merge_debevec = cv.createMergeDebevec()
    hdr = merge_debevec.process(images, times, response)
    hdr = hdr/np.max(hdr)

    # Objective
    metrics = [
        pyiqa.create_metric("brisque"),
        pyiqa.create_metric("ilniqe"),
        pyiqa.create_metric("niqe"),
        pyiqa.create_metric("piqe"),
        pyiqa.create_metric("arniqa"),
        pyiqa.create_metric("nima"),
    ]

    n_detail_layers = 3
    def objectives(params):
        params_mapped = map_parameters(params, n_detail_layers)
        out           = pipeline(hdr.copy(), n_detail_layers, params_mapped)
        out_rgb       = out[...,::-1].copy()
        out_tensor    = torch.tensor(out_rgb, dtype=torch.float32).permute([2,1,0]).unsqueeze(0)
        return list(map(lambda metric : metric(out_tensor).item(), metrics))

    def visualize(params):
        params_mapped = map_parameters(params, n_detail_layers)
        out           = pipeline(hdr, n_detail_layers, params_mapped)
        plt.subplot(121), plt.imshow(hdr[...,::-1])
        plt.subplot(122), plt.imshow(out[...,::-1])
        plt.show()

    params        = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    params_mapped = map_parameters(params, n_detail_layers)
    print(params_mapped)
    print(objectives(params))
    print(visualize(params))

if __name__ == "__main__":
    main()
    # OUT: 
    # {'gf_radiuses': array([18, 18, 18]), 'gf_epsilons': array([0.505, 0.505, 0.505]), 'detail_gains': array([0.75, 0.75, 0.75]), 'base_gain': np.float64(0.5), 'color_gain': np.float64(0.5), 'invgamma': np.float64(3.0), 'gain': np.float64(1.1)}
    # [4.03302001953125, 21.749099285307178, 3.0919498138420756, 20.80258560180664, 0.580994188785553, 5.10910701751709]
    # None

