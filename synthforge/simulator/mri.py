"""Procedural MRI-like phantoms (2D slices and small 3D volumes).

These are parametric ellipsoid phantoms with T1/T2/FLAIR-like contrast, bias fields, Rician noise
and masks. They are explicitly **not** human anatomy and carry no clinical meaning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from synthforge.simulator.common import rng

CONTRASTS = ["T1", "T2", "FLAIR"]


def _ellipse(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float, angle: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    yy = (yy - cy) / max(ry, 1e-6)
    xx = (xx - cx) / max(rx, 1e-6)
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * xx + sa * yy
    yr = -sa * xx + ca * yy
    return (xr**2 + yr**2) <= 1.0


def _ellipsoid(shape: tuple[int, int, int], center, radii, angle: float = 0.0) -> np.ndarray:
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    z = (zz - center[0]) / max(radii[0], 1e-6)
    y = (yy - center[1]) / max(radii[1], 1e-6)
    x = (xx - center[2]) / max(radii[2], 1e-6)
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return (xr**2 + yr**2 + z**2) <= 1.0


def _contrast_image(
    shape: tuple[int, int],
    contrast: str,
    lesion: bool,
    r: np.random.Generator,
    lesion_center: tuple[float, float] | None = None,
    lesion_radii: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    cy, cx = h / 2 + r.normal(0, 1.5), w / 2 + r.normal(0, 1.5)
    head = _ellipse(shape, cy, cx, h * 0.42, w * 0.36)
    skull = _ellipse(shape, cy, cx, h * 0.435, w * 0.375) & ~head
    wm = _ellipse(shape, cy, cx, h * 0.31, w * 0.26)
    gm = _ellipse(shape, cy, cx, h * 0.40, w * 0.335) & ~wm
    ventricle = _ellipse(shape, cy - h * 0.02, cx, h * 0.11, w * 0.075)
    ventricle |= _ellipse(shape, cy - h * 0.02, cx - w * 0.10, h * 0.08, w * 0.045)
    ventricle |= _ellipse(shape, cy - h * 0.02, cx + w * 0.10, h * 0.08, w * 0.045)

    if contrast == "T1":
        vals = {"bg": 0.02, "skull": 0.10, "gm": 0.42, "wm": 0.68, "csf": 0.08, "lesion": 0.82}
    elif contrast == "T2":
        vals = {"bg": 0.01, "skull": 0.06, "gm": 0.52, "wm": 0.36, "csf": 0.92, "lesion": 0.88}
    else:  # FLAIR
        vals = {"bg": 0.01, "skull": 0.05, "gm": 0.55, "wm": 0.42, "csf": 0.05, "lesion": 0.90}

    img = np.full(shape, vals["bg"], dtype=np.float32)
    img[skull] = vals["skull"]
    img[gm] = vals["gm"]
    img[wm] = vals["wm"]
    img[ventricle] = vals["csf"]
    mask = np.zeros(shape, dtype=np.float32)
    if lesion:
        lc = lesion_center or (cy + r.uniform(-0.12, 0.12) * h, cx + r.uniform(-0.14, 0.14) * w)
        lr = lesion_radii or (r.uniform(0.06, 0.14) * h, r.uniform(0.06, 0.14) * w)
        lesion_mask = _ellipse(shape, lc[0], lc[1], lr[0], lr[1], angle=r.uniform(0, np.pi))
        ring = _ellipse(shape, lc[0], lc[1], lr[0] * 1.18, lr[1] * 1.18, angle=0.0) & ~lesion_mask
        img[ring] = np.clip(vals["lesion"] * 0.72, 0, 1)
        img[lesion_mask] = np.clip(vals["lesion"] + r.normal(0, 0.03), 0, 1)
        # necrotic core for larger lesions
        if min(lr) > 0.09 * h:
            core = _ellipse(shape, lc[0], lc[1], lr[0] * 0.35, lr[1] * 0.35)
            img[core] = vals["csf"] * 0.8 + 0.15
        mask[lesion_mask] = 1.0
    # smooth tissue boundaries
    for k, v in vals.items():
        region = {"bg": ~head, "skull": skull, "gm": gm, "wm": wm, "csf": ventricle, "lesion": mask > 0}.get(k)
        if region is not None and k in ("gm", "wm"):
            img[region] = v
    img = gaussian_filter(img, sigma=0.8)
    return np.clip(img, 0, 1).astype(np.float32), mask


def make_phantom_2d(
    resolution: int = 64,
    *,
    contrast: str = "T1",
    lesion: bool = True,
    seed: int = 0,
    rician_snr: float = 26.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    r = rng(seed)
    img, mask = _contrast_image((resolution, resolution), contrast, lesion, r)
    # smooth multiplicative bias field
    yy, xx = np.mgrid[0:resolution, 0:resolution] / max(resolution - 1, 1)
    coeffs = r.normal(0, 0.10, 6)
    bias = 1 + coeffs[0] * (xx - 0.5) + coeffs[1] * (yy - 0.5) + coeffs[2] * (xx - 0.5) * (yy - 0.5) + coeffs[3] * (xx - 0.5) ** 2 + coeffs[4] * (yy - 0.5) ** 2
    img = img * np.clip(bias, 0.6, 1.4)
    # Rician noise
    sigma = float(np.clip(np.mean(img[img > 0.1]) / rician_snr, 0.005, 0.08))
    noise = (r.normal(0, sigma, img.shape) + 1j * r.normal(0, sigma, img.shape)) / np.sqrt(2)
    img = np.abs(img + noise).astype(np.float32)
    # percentile normalisation to [0,1]
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    meta = {"contrast": contrast, "lesion": int(lesion), "seed": seed, "sigma": sigma, "synthetic": True}
    return img, mask, meta


def make_phantom_3d(resolution: int = 32, *, seed: int = 0, lesion: bool = True) -> tuple[np.ndarray, np.ndarray]:
    r = rng(seed)
    shape = (resolution, resolution, resolution)
    c = (resolution / 2, resolution / 2, resolution / 2)
    head = _ellipsoid(shape, c, (resolution * 0.42, resolution * 0.36, resolution * 0.40))
    wm = _ellipsoid(shape, c, (resolution * 0.30, resolution * 0.25, resolution * 0.29))
    vent = _ellipsoid(shape, (c[0] - 1, c[1], c[2]), (resolution * 0.08, resolution * 0.12, resolution * 0.06))
    img = np.zeros(shape, dtype=np.float32)
    img[head] = 0.42
    img[wm] = 0.65
    img[vent] = 0.08
    mask = np.zeros(shape, dtype=np.float32)
    if lesion:
        lc = (c[0] + r.uniform(-4, 4), c[1] + r.uniform(-4, 4), c[2] + r.uniform(-4, 4))
        m = _ellipsoid(shape, lc, (r.uniform(2.0, 4.0), r.uniform(2.0, 4.0), r.uniform(2.0, 4.0)))
        img[m] = 0.85
        mask[m] = 1.0
    img = gaussian_filter(img, sigma=0.6)
    sigma = 0.025
    noise = (r.normal(0, sigma, shape) + 1j * r.normal(0, sigma, shape)) / np.sqrt(2)
    img = np.clip(np.abs(img + noise), 0, 1).astype(np.float32)
    return img, mask


def simulate_mri_cohort(
    n: int = 80, seed: int = 51, resolution: int = 64, *, include_3d: bool = False, n3d: int = 8
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    r = rng(seed)
    contrasts = r.choice(CONTRASTS, n, p=[0.4, 0.3, 0.3])
    lesion = r.random(n) < 0.42
    images = np.zeros((n, resolution, resolution), dtype=np.float32)
    masks = np.zeros((n, resolution, resolution), dtype=np.float32)
    rows = []
    for i in range(n):
        img, mask, meta = make_phantom_2d(
            resolution, contrast=str(contrasts[i]), lesion=bool(lesion[i]), seed=seed * 1000 + i
        )
        images[i] = img
        masks[i] = mask
        rows.append(
            {
                "study_id": f"MRI{i:05d}",
                "patient_id": f"P{i // 2 + 1:05d}",
                "contrast": str(contrasts[i]),
                "lesion": int(lesion[i]),
                "slice_index": 0,
                "age": int(np.clip(r.normal(55, 18), 18, 95)),
                "source": "seeded_ellipsoid_phantom",
            }
        )
    labels = pd.DataFrame(rows)
    images3d = masks3d = None
    labels3d = None
    if include_3d:
        images3d = np.zeros((n3d, resolution // 2, resolution // 2, resolution // 2), dtype=np.float32)
        masks3d = np.zeros_like(images3d)
        rows3d = []
        for i in range(n3d):
            img, mask = make_phantom_3d(resolution // 2, seed=seed * 2000 + i, lesion=bool(lesion[i]))
            images3d[i], masks3d[i] = img, mask
            rows3d.append(
                {
                    "study_id": f"MRI3D{i:04d}",
                    "patient_id": f"P{i:05d}",
                    "contrast": "T1",
                    "lesion": int(lesion[i]),
                    "slice_index": -1,
                    "age": int(np.clip(r.normal(55, 18), 18, 95)),
                    "source": "seeded_ellipsoid_phantom_3d",
                }
            )
        labels3d = pd.DataFrame(rows3d)
    return images, masks, labels, labels3d, images3d, masks3d


def write_mri_nifti(
    root: str | Path,
    n: int = 24,
    seed: int = 51,
    resolution: int = 64,
    *,
    include_3d: bool = False,
) -> Path:
    import nibabel as nib

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images, masks, labels, labels3d, images3d, masks3d = simulate_mri_cohort(
        n=n, seed=seed, resolution=resolution, include_3d=include_3d
    )
    rows = []
    for i in range(n):
        study = labels.iloc[i]["study_id"]
        probe = nib.Nifti1Image(images[i], affine=np.diag([1.0, 1.0, 3.0, 1.0]))
        probe.header.set_data_dtype(np.float32)
        probe.header["descrip"] = b"SYNTHFORGE PHANTOM - NOT ANATOMY"
        nib.save(probe, str(root / f"{study}_{labels.iloc[i]['contrast']}.nii.gz"))
        mask_img = nib.Nifti1Image(masks[i], affine=np.diag([1.0, 1.0, 3.0, 1.0]))
        nib.save(mask_img, str(root / f"{study}_{labels.iloc[i]['contrast']}_seg.nii.gz"))
        rows.append(
            {
                **labels.iloc[i].to_dict(),
                "file": f"{study}_{labels.iloc[i]['contrast']}.nii.gz",
                "mask_file": f"{study}_{labels.iloc[i]['contrast']}_seg.nii.gz",
                "voxel_spacing": "1.0,1.0,3.0",
            }
        )
    if include_3d and images3d is not None and labels3d is not None and masks3d is not None:
        for i in range(len(images3d)):
            study = labels3d.iloc[i]["study_id"]
            img = nib.Nifti1Image(images3d[i], affine=np.diag([1.0, 1.0, 2.0, 1.0]))
            nib.save(img, str(root / f"{study}_T1.nii.gz"))
            nib.save(
                nib.Nifti1Image(masks3d[i], affine=np.diag([1.0, 1.0, 2.0, 1.0])),
                str(root / f"{study}_T1_seg.nii.gz"),
            )
            rows.append({**labels3d.iloc[i].to_dict(), "file": f"{study}_T1.nii.gz", "mask_file": f"{study}_T1_seg.nii.gz", "voxel_spacing": "1.0,1.0,2.0"})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    (root / "README.txt").write_text(
        "SYNTHETIC SOURCE: parametric ellipsoid MRI phantoms. Not human anatomy. No PHI.\n",
        encoding="utf-8",
    )
    return root


def write_mri_dicom(root: str | Path, n_studies: int = 2, seed: int = 61, resolution: int = 64) -> Path:
    """Write small DICOM series with clearly fake PHI tags (used to test de-identification)."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images, masks, labels, _, _, _ = simulate_mri_cohort(n=n_studies, seed=seed, resolution=resolution)
    report = []
    for i in range(n_studies):
        study_dir = root / f"study_{i:03d}"
        study_dir.mkdir(exist_ok=True)
        study_uid = generate_uid()
        series_uid = generate_uid()
        for z in range(3):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.ImplementationClassUID = generate_uid()
            ds = FileDataset(str(study_dir / f"slice_{z:03d}.dcm"), Dataset(), file_meta=meta, preamble=b"\0" * 128)
            ds.PatientName = f"PHANTOM^SYNTHETIC_{i}"
            ds.PatientID = f"SYNTH-{i:04d}"
            ds.PatientBirthDate = "19800101"
            ds.PatientAddress = "1 Fake Street, Testville"
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.Modality = "MR"
            ds.StudyDescription = "SYNTHETIC PHANTOM"
            ds.SeriesDescription = str(labels.iloc[i]["contrast"])
            ds.Manufacturer = "SynthForgeSimulator"
            ds.InstanceNumber = z + 1
            ds.SliceLocation = float(z * 3.0)
            ds.PixelSpacing = [1.0, 1.0]
            ds.SliceThickness = 3.0
            ds.Rows, ds.Columns = resolution, resolution
            arr = (np.clip(images[i] * 255, 0, 255)).astype(np.uint16)
            arr = np.roll(arr, z, axis=0)
            ds.PixelData = arr.tobytes()
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.file_meta = meta
            ds.save_as(str(study_dir / f"slice_{z:03d}.dcm"), enforce_file_format=True)
        report.append({"study_dir": study_dir.name, "fake_phi": True, "synthetic": True})
    (root / "deid_note.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return root
