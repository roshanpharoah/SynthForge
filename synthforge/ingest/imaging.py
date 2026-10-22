"""Medical imaging ingestion: NIfTI and DICOM, with PHI stripping and de-identification report."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from synthforge.core.errors import IngestError
from synthforge.core.types import Dataset, ImageCohort
from synthforge.ingest.profile import profile_dataset

PHI_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientComments",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "DeviceSerialNumber",
    "StudyID",
    "AccessionNumber",
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "RequestedProcedureDescription",
    "AdmissionID",
]

DATE_PATTERN = re.compile(r"\b(19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")


def _resize(img: np.ndarray, target: tuple[int, ...], *, mask: bool = False) -> np.ndarray:
    if img.shape == target:
        return img.astype(np.float32)
    factors = [t / max(s, 1) for t, s in zip(target, img.shape, strict=False)]
    out = zoom(img.astype(np.float32), factors, order=0 if mask else 1)
    # pad/crop to exact target after rounding
    slices = tuple(slice(0, t) for t in target)
    pad = [(0, max(0, t - o)) for t, o in zip(target, out.shape, strict=False)]
    if any(p[1] > 0 for p in pad):
        out = np.pad(out, pad, mode="constant")
    return out[slices].astype(np.float32)


def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    out = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _load_labels(path: Path | None, root: Path) -> pd.DataFrame | None:
    candidates = [path, root / "labels.csv", root / "metadata.csv"] if path else [root / "labels.csv"]
    for c in candidates:
        if c and Path(c).exists():
            try:
                df = pd.read_csv(c)
            except Exception as exc:
                raise IngestError(f"could not read imaging labels: {exc}") from exc
            if "file" in df.columns:
                df["_key"] = df["file"].astype(str).str.replace(r"\.nii(\.gz)?$", "", regex=True)
            return df
    return None


def _nifti_to_slices(
    path: Path,
    *,
    target_resolution: int,
    volume_mode: bool,
    slices_per_volume: int,
    mask_path: Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray | None], dict[str, Any]]:
    import nibabel as nib

    try:
        img = nib.load(str(path))
    except Exception as exc:
        raise IngestError(
            f"could not read NIfTI {path.name}: {exc}",
            hint="Verify the file is an uncompressed or gzip-compressed NIfTI volume.",
        ) from exc
    canonical = nib.as_closest_canonical(img)
    data = np.asarray(canonical.get_fdata(dtype=np.float32))
    if data.ndim == 4:
        raise IngestError(
            f"{path.name} is 4D",
            hint="Pass a single 3D volume or extract time frames before ingestion.",
        )
    if data.ndim not in (2, 3):
        raise IngestError(f"unsupported NIfTI dimensionality {data.ndim}", hint="Use a 2D or 3D volume.")
    header = canonical.header
    header_strings = []
    for key in ("descrip", "aux_file", "db_name", "intent_name"):
        try:
            val = header[key]
            if isinstance(val, bytes):
                val = val.decode("latin-1", errors="ignore")
            if val and str(val).strip():
                header_strings.append(str(val).strip())
        except Exception:
            pass
    deid = {
        "nifti_header_fields_dropped": header_strings,
        "date_like_header_strings": [s for s in header_strings if DATE_PATTERN.search(s)],
        "pixel_data_only": True,
        "note": "Only voxel arrays and spacing are persisted; NIfTI headers are not copied to artifacts.",
    }
    mask_data = None
    if mask_path and mask_path.exists():
        try:
            mask_img = nib.as_closest_canonical(nib.load(str(mask_path)))
            mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32)) > 0.5
        except Exception as exc:
            raise IngestError(f"could not read mask {mask_path.name}: {exc}") from exc
        if mask_data.shape != data.shape:
            raise IngestError(
                f"mask shape {mask_data.shape} does not match image {data.shape}",
                hint="Resample the mask to the image grid before ingestion.",
            )
    data = _normalize_volume(data)
    if data.ndim == 2:
        return [_resize(data, (target_resolution, target_resolution))], [
            None if mask_data is None else _resize(mask_data.astype(np.float32), (target_resolution, target_resolution), mask=True)
        ], deid
    if volume_mode:
        depth = max(8, min(32, data.shape[2]))
        img_r = _resize(data, (target_resolution, target_resolution, depth))
        mask_r = None if mask_data is None else _resize(mask_data.astype(np.float32), (target_resolution, target_resolution, depth), mask=True)
        return [img_r], [mask_r], deid
    # pick informative axial slices
    z = data.shape[2]
    energies = np.array([float(np.mean(np.abs(data[:, :, k]))) for k in range(z)])
    order = np.argsort(energies)[::-1]
    chosen = sorted(order[: min(slices_per_volume, z)].tolist())
    imgs = []
    masks = []
    for k in chosen:
        imgs.append(_resize(data[:, :, k], (target_resolution, target_resolution)))
        if mask_data is not None:
            masks.append(
                _resize(mask_data[:, :, k].astype(np.float32), (target_resolution, target_resolution), mask=True)
            )
        else:
            masks.append(None)
    return imgs, masks, deid


def _find_niftis(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            if "_seg" in p.name or p.name.endswith("_mask.nii.gz"):
                continue
            files.append(p)
    return files


def _dicom_series_dirs(root: Path) -> list[Path]:
    dirs = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".dcm", ".dicom"):
            dirs.add(p.parent)
        if p.is_file() and not p.suffix and p.stat().st_size > 132:
            # extensionless DICOM is common; inspect preamble
            try:
                with open(p, "rb") as fh:
                    if fh.read(132)[128:132] == b"DICM":
                        dirs.add(p.parent)
            except OSError:
                pass
    return sorted(dirs)


def _read_dicom_series(
    folder: Path, *, target_resolution: int, volume_mode: bool, slices_per_volume: int
) -> tuple[list[np.ndarray], list[np.ndarray | None], dict[str, Any], tuple[float, ...]]:
    import pydicom

    files = sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
            and (p.suffix.lower() in (".dcm", ".dicom") or (p.suffix == "" and p.stat().st_size > 132))
        ]
    )
    slices = []
    found_tags: dict[str, int] = {}
    removed_private = 0
    spacings = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except Exception as exc:
            raise IngestError(
                f"could not read DICOM {f.name}: {exc}",
                hint="Check transfer syntax support and that the file is not encrypted.",
            ) from exc
        for tag in PHI_TAGS:
            if hasattr(ds, tag) and getattr(ds, tag) not in (None, ""):
                found_tags[tag] = found_tags.get(tag, 0) + 1
        if hasattr(ds, "remove_private_tags"):
            ds.remove_private_tags()
            removed_private += 1
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise IngestError(
                f"DICOM pixel decoding failed for {f.name}: {exc}",
                hint="Compressed transfer syntaxes need a codec (pylibjpeg) or conversion to uncompressed DICOM.",
            ) from exc
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        z = float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", len(slices))))
        slices.append((z, arr))
        if hasattr(ds, "PixelSpacing") and len(ds.PixelSpacing) == 2:
            spacings.append((float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])))
    slices.sort(key=lambda x: x[0])
    arrays = [a for _, a in slices]
    if not arrays:
        raise IngestError(f"no DICOM slices found in {folder.name}")
    shape = arrays[0].shape
    same = [a for a in arrays if a.shape == shape]
    vol = np.stack(same, axis=0)
    spacing = (
        (float(getattr(pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True), "SliceThickness", 1.0) or 1.0),)
        + (spacings[0] if spacings else (1.0, 1.0))
    )
    vol = _normalize_volume(vol)
    if volume_mode:
        depth = max(8, min(32, vol.shape[0]))
        return (
            [_resize(vol, (target_resolution, target_resolution, depth))],
            [None],
            {
                "phi_tags_found": found_tags,
                "private_tags_removed": removed_private,
                "pixel_data_only": True,
                "dicom_header_dropped": True,
                "note": "DICOM headers and all PHI tags were stripped; only pixel arrays and spacing are kept.",
            },
            spacing,
        )
    z = vol.shape[0]
    chosen = sorted(np.argsort([float(np.mean(np.abs(vol[k]))) for k in range(z)])[::-1][: min(slices_per_volume, z)].tolist())
    return (
        [_resize(vol[k], (target_resolution, target_resolution)) for k in chosen],
        [None for _ in chosen],
        {
            "phi_tags_found": found_tags,
            "private_tags_removed": removed_private,
            "pixel_data_only": True,
            "dicom_header_dropped": True,
            "note": "DICOM headers and all PHI tags were stripped; only pixel arrays and spacing are kept.",
        },
        spacing,
    )


def load_imaging(
    path: str | Path,
    *,
    name: str | None = None,
    mask: str | Path | None = None,
    labels_path: str | Path | None = None,
    target_resolution: int = 64,
    volume_mode: bool = False,
    slices_per_volume: int = 3,
    max_volumes: int | None = None,
    profile: bool = True,
    synthetic_source: bool = False,
) -> Dataset:
    p = Path(path).resolve()
    if not p.exists():
        raise IngestError(f"imaging path not found: {p}", hint="Pass a NIfTI file, DICOM directory, or cohort folder.")
    labels = _load_labels(Path(labels_path) if labels_path else None, p if p.is_dir() else p.parent)
    images: list[np.ndarray] = []
    masks: list[np.ndarray | None] = []
    rows: list[dict[str, Any]] = []
    deid_reports: list[dict[str, Any]] = []
    spacings: list[tuple[float, ...]] = []
    warnings: list[str] = []
    if p.is_file():
        units = [(p, Path(mask) if mask else None, p.stem)]
    else:
        niftis = _find_niftis(p)
        if niftis:
            units = []
            for f in niftis:
                key = f.name.replace(".nii.gz", "").replace(".nii", "")
                m = None
                if mask:
                    m = Path(mask)
                else:
                    candidate = f.with_name(key + "_seg.nii.gz")
                    if candidate.exists():
                        m = candidate
                units.append((f, m, key))
        else:
            dcm_dirs = _dicom_series_dirs(p)
            if not dcm_dirs:
                raise IngestError(
                    "no NIfTI volumes or DICOM files found",
                    hint="Use .nii/.nii.gz or a directory containing .dcm slices.",
                )
            units = [(d, None, d.name) for d in dcm_dirs]
    if max_volumes:
        units = units[:max_volumes]
    for f, m, key in units:
        if f.is_dir():
            vol_imgs, vol_masks, deid, spacing = _read_dicom_series(
                f, target_resolution=target_resolution, volume_mode=volume_mode, slices_per_volume=slices_per_volume
            )
        else:
            vol_imgs, vol_masks, deid = _nifti_to_slices(
                f,
                target_resolution=target_resolution,
                volume_mode=volume_mode,
                slices_per_volume=slices_per_volume,
                mask_path=m,
            )
            try:
                import nibabel as nib

                zooms = nib.load(str(f)).header.get_zooms()
                spacing = tuple(float(v) for v in zooms[:3])
            except Exception:
                spacing = ()
        deid["file"] = f.name
        deid_reports.append(deid)
        spacings.append(spacing)
        lab = {}
        if labels is not None:
            match = labels[labels["_key"] == key] if "_key" in labels.columns else labels.iloc[0:0]
            if len(match):
                lab = {k: v for k, v in match.iloc[0].to_dict().items() if not str(k).startswith("_")}
        if not lab:
            contrast = "T1" if "T1" in f.name else ("T2" if "T2" in f.name else ("FLAIR" if "FLAIR" in f.name else "unknown"))
            lab = {"study_id": key, "contrast": contrast, "lesion": 0}
        for j, (img, msk) in enumerate(zip(vol_imgs, vol_masks, strict=False)):
            images.append(img)
            masks.append(msk)
            rows.append({**lab, "slice_index": j if not volume_mode else -1, "source_file": f.name})
    arr = np.stack(images)
    mask_arr = None if all(x is None for x in masks) else np.stack([np.zeros_like(images[0]) if x is None else x for x in masks])
    labels_df = pd.DataFrame(rows)
    if "synthetic_source" not in labels_df.columns:
        labels_df["synthetic_source"] = synthetic_source
    deid_agg = {
        "files_processed": len(units),
        "records": len(images),
        "phi_tags_found": _merge_counts(deid_reports),
        "pixel_data_only": True,
        "header_metadata_dropped": True,
        "details": deid_reports,
        "note": "Only de-identified pixel arrays, spacing, and supplied condition labels are persisted.",
    }
    if deid_agg["phi_tags_found"]:
        warnings.append(
            "DICOM PHI tags were present and stripped: " + ", ".join(sorted(deid_agg["phi_tags_found"]))
        )
    cohort = ImageCohort(
        images=arr,
        masks=mask_arr,
        labels=labels_df,
        spacing=spacings[0] if spacings else (),
        metadata={
            "volume_mode": volume_mode,
            "synthetic": synthetic_source,
            "source_format": "dicom" if units and units[0][0].is_dir() else "nifti",
        },
        deid_report=deid_agg,
        source_paths=[str(u[0]) for u in units],
    )
    dataset = Dataset(
        name=name or p.stem,
        kind="imaging",
        images=cohort,
        source={"type": "imaging", "path": str(p), "format": cohort.metadata["source_format"]},
        warnings=warnings,
        provenance={
            "loader": "synthforge.ingest.imaging",
            "synthetic_source": synthetic_source,
            "deidentification": deid_agg,
        },
    )
    dataset.profile = profile_dataset(dataset) if profile else {}
    return dataset


def _merge_counts(reports: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in reports:
        for k, v in (r.get("phi_tags_found") or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out
