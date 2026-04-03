import numpy as np

# import scipy.signal
from collections import Counter
from scipy.ndimage import maximum_filter, gaussian_laplace, gaussian_filter, uniform_filter, binary_dilation, distance_transform_edt, label,binary_closing


import SimpleITK as sitk
from pathlib import Path
import os
import copy
from tqdm import tqdm, trange
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import rich


def create_progress(bar_width=40):
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )
    return progress



def set_properties(new_image, image):
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetDirection(image.GetDirection())
    return new_image


# ========= No-label helpers=========
def normalize01(a, eps=1e-6):
    a = a.astype(np.float32)
    mn, mx = float(np.min(a)), float(np.max(a))
    if mx - mn < eps: 
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)

def safe_percentile(a, q):
    a = a[np.isfinite(a)]
    if a.size == 0: 
        return 0.0
    return float(np.percentile(a, q))

def zscore_in_mask(a, mask, p_low=0.5, p_high=99.5, eps=1e-6):

    vals = a[mask > 0]
    if vals.size < 50:   # 太少就退化到全局
        vals = a
    lo = safe_percentile(vals, p_low); hi = safe_percentile(vals, p_high)
    a_clip = np.clip(a, lo, hi)
    mu = float(vals.mean()); sd = float(vals.std()) + eps
    return (a_clip - mu) / sd

def soft_weights_from_roi(roi, decay=4.0, extra_dilate=0):

    if extra_dilate > 0:
        roi = binary_dilation(roi.astype(bool), iterations=int(extra_dilate))
    dist = distance_transform_edt(~roi.astype(bool))
    if dist.max() > 0:
        dist = dist / (dist.max() + 1e-6)
    w = np.exp(-float(decay) * dist).astype(np.float32)
    w[roi.astype(bool)] = 1.0
    return w

def auto_tissue_mask(I):

    lo = safe_percentile(I, 0.5); hi = safe_percentile(I, 99.5)
    I2 = np.clip(I, lo, hi).astype(np.float32)
    itk = sitk.GetImageFromArray(I2)
    otsu = sitk.OtsuThreshold(itk, 0, 1)  
    m = sitk.GetArrayFromImage(otsu).astype(np.uint8)


    cc, n = label(m)
    if n > 0:
        counts = np.bincount(cc.ravel())
        counts[0] = 0
        k = int(np.argmax(counts))
        m = (cc == k).astype(np.uint8)
    m = binary_dilation(m, iterations=2).astype(np.uint8)
    return m

def saliency_map(I, tissue_mask, sigmas=(1.0, 2.0, 3.0), tz=1.5, w_z=0.45, w_log=0.35, w_var=0.20):

    Z = zscore_in_mask(I, tissue_mask, 0.5, 99.5)
    z_abs = np.abs(Z)
    z_sal = np.clip(z_abs - float(tz), 0, None)
    z_sal = normalize01(z_sal)

    log_sum = np.zeros_like(I, dtype=np.float32)
    for s in sigmas:
        log_sum += np.abs(gaussian_laplace(I.astype(np.float32), sigma=float(s)))
    log_sum = normalize01(log_sum)

    I_f = I.astype(np.float32)

    if I.ndim == 3:
        size_mean = (5, 5, 1)  
        mu  = uniform_filter(I_f, size=size_mean, mode="nearest")
        mu2 = uniform_filter(I_f * I_f, size=size_mean, mode="nearest")
    else:
        size_mean = 5
        mu  = uniform_filter(I_f, size=size_mean, mode="nearest")
        mu2 = uniform_filter(I_f * I_f, size=size_mean, mode="nearest")

    vmap = mu2 - mu * mu
    vmap[vmap < 0] = 0.0    
    vmap = normalize01(vmap)


    S = w_z * z_sal + w_log * log_sum + w_var * vmap
    S *= tissue_mask  
    S = normalize01(S)
    return S

def build_pseudo_roi(S, tissue_mask, seed_top_percent=0.8, min_vox=500, dilate_iter=5):

    S_in = S[tissue_mask > 0]
    if S_in.size == 0:
        return np.zeros_like(S, dtype=np.uint8)
    thr = safe_percentile(S_in, 100.0 - float(seed_top_percent))  
    seeds = (S >= thr) & (tissue_mask > 0)


    cc, n = label(seeds.astype(np.uint8))
    keep = np.zeros_like(S, dtype=bool)
    if n > 0:
        areas = np.bincount(cc.ravel())
        order = np.argsort(areas)[::-1]  
        for idx in order[:8]:  
            if idx == 0: 
                continue
            if areas[idx] >= int(min_vox):
                keep |= (cc == idx)
    if not keep.any():

        if n > 0:
            keep = (cc == int(np.argmax(np.bincount(cc.ravel())[1:]) + 1))
        else:
            keep = seeds

    roi = binary_dilation(keep, iterations=int(dilate_iter)).astype(np.uint8)
    return roi
# ============================================

def keep_largest_components(mask, k=3, min_vox=5000, close_iter=1, dilate_iter=2):

    cc, n = label(mask.astype(np.uint8))
    if n <= 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    order = np.argsort(sizes)[::-1]  
    keep = np.zeros_like(mask, dtype=bool)
    for idx in order[:k]:
        if sizes[idx] >= int(min_vox):
            keep |= (cc == idx)
    mask = keep.astype(np.uint8)
    if close_iter > 0:
        mask = binary_closing(mask, iterations=int(close_iter)).astype(np.uint8)
    if dilate_iter > 0:
        mask = binary_dilation(mask, iterations=int(dilate_iter)).astype(np.uint8)
    return mask

def counter(label_array):
    count = Counter(label_array.flatten())
    zero_num = count[0]
    one_num = count[1]
    ratio = one_num / (zero_num + one_num)
    rich.print( ratio)
    return ratio


def rescale(data_path, save_path):
    data_list = sorted(Path(data_path).glob("*.nii*"))
    os.makedirs(save_path, exist_ok=True)
    progress = create_progress()
    for i in progress.track(data_list, total=len(data_list), description="Processing Data"):
        progress.start()
        image = sitk.ReadImage(i)
        image = sitk.Cast(image, sitk.sitkUInt16)

        # rescale image
        rescale_filter = sitk.RescaleIntensityImageFilter()
        rescale_filter.SetOutputMaximum(255)
        rescale_filter.SetOutputMinimum(0)
        image = rescale_filter.Execute(image)
        print(save_path + i.name)

        sitk.WriteImage(image, save_path + i.name)


def create_tumor_local_nolabel(data_path, save_path,
                               slice_wise=False,
                               seed_top_percent=0.8,   
                               dilate_iter=5,          
                               decay_k=4.0,            
                               p_low=1.0, p_high=99.0  
                               ):

    data_list = sorted(Path(data_path).glob("*.nii*"))
    os.makedirs(save_path, exist_ok=True)

    for img_p in data_list:
        image = sitk.ReadImage(str(img_p))
        I = sitk.GetArrayFromImage(image).astype(np.float32)   

        tissue = auto_tissue_mask(I)

        S = saliency_map(I, tissue_mask=tissue)
        if slice_wise and I.ndim == 3:
            roi = np.zeros_like(I, dtype=np.uint8)
            for z in range(I.shape[0]):
                S_z = S[z]; T_z = tissue[z]
                roi[z] = build_pseudo_roi(S_z, T_z, seed_top_percent=seed_top_percent, min_vox=200, dilate_iter=dilate_iter)
        else:
            roi = build_pseudo_roi(S, tissue, seed_top_percent=seed_top_percent, min_vox=500, dilate_iter=dilate_iter)

        roi_big = keep_largest_components(roi, k=3, min_vox=5000, close_iter=1, dilate_iter=2)

        vox_roi = float(np.count_nonzero(roi))
        vox_tis = float(np.count_nonzero(tissue)) + 1e-6
        r = vox_roi / vox_tis  
        bg_mask = (tissue > 0) & (roi == 0)
        bg_vals = I[bg_mask]
        if bg_vals.size == 0:  
            bg_vals = I.flatten()
        T_bg = max(safe_percentile(bg_vals, 90.0), safe_percentile(bg_vals, (1.0 - r) * 100.0))
        alpha = float(np.clip(0.3 + 0.4 * (1.0 - r), 0.3, 0.9))

        I_bg = I.copy()
        I_bg[bg_mask] = np.minimum(I_bg[bg_mask], T_bg)
        I_bg[bg_mask] = (1.0 - alpha) * I_bg[bg_mask]

        I_loc = zscore_in_mask(I, mask=roi, p_low=float(p_low), p_high=float(p_high))

        W = soft_weights_from_roi(roi.astype(bool), decay=float(decay_k), extra_dilate=0)
        I_bg_n  = normalize01(I_bg)
        I_loc_n = normalize01(I_loc)
        I_enh   = W * I_loc_n + (1.0 - W) * I_bg_n

        I_orig_n = normalize01(I)
        I_dark_n = normalize01(I_bg)  

        # out_enh = sitk.GetImageFromArray((I_enh * 255.0).astype(np.uint8))
        # out_enh = set_properties(out_enh, image)
        # sitk.WriteImage(out_enh, os.path.join(save_path, img_p.name.replace(".nii", "_enh.nii")))

        out_dark = sitk.GetImageFromArray((I_dark_n * 255.0).astype(np.uint8))
        out_dark = set_properties(out_dark, image)
        sitk.WriteImage(out_dark, os.path.join(save_path, img_p.name.replace(".nii", "_dark.nii")))

        # out_orig = sitk.GetImageFromArray((I_orig_n * 255.0).astype(np.uint8))
        # out_orig = set_properties(out_orig, image)
        # sitk.WriteImage(out_orig, os.path.join(save_path, img_p.name.replace(".nii", "_orig_n.nii")))



if __name__ == "__main__":
 

# --------- No-label tumor-local pipeline ----------
    data_path = " " 
    save_path = " " 
    os.makedirs(save_path, exist_ok=True)

    create_tumor_local_nolabel(
        data_path=data_path,
        save_path=save_path,
        slice_wise=False,      
        seed_top_percent=0.5,   
        dilate_iter=5,
        decay_k=4.0,
        p_low=1.0, p_high=99.0
    )
