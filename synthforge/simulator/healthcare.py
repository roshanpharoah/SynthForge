"""Synthetic healthcare relational data with linked ECG/MRI metadata.

All records are simulated. Conditions are causally generated from patient attributes so cross-modal
correlations exist and can be tested; they are not medically meaningful diagnoses.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from synthforge.simulator.common import add_missing, random_dates, rng, synthetic_name

CONDITIONS = ["healthy", "diabetes", "hypertension", "cardiac", "neuro", "respiratory"]
ICD = {
    "diabetes": ("E11.9", "Type 2 diabetes mellitus (synthetic label)"),
    "hypertension": ("I10", "Essential hypertension (synthetic label)"),
    "cardiac": ("I50.9", "Heart failure (synthetic label)"),
    "neuro": ("G40.9", "Epilepsy (synthetic label)"),
    "respiratory": ("J44.9", "COPD (synthetic label)"),
}
DRUGS = {
    "diabetes": ["metformin", "insulin_glargine"],
    "hypertension": ["lisinopril", "amlodipine"],
    "cardiac": ["furosemide", "carvedilol"],
    "neuro": ["levetiracetam"],
    "respiratory": ["albuterol", "tiotropium"],
}


def simulate_healthcare(seed: int = 13, n_patients: int = 240, n_encounters: int = 700) -> dict[str, pd.DataFrame]:
    r = rng(seed)
    pids = [f"P{i:05d}" for i in range(1, n_patients + 1)]
    age = np.clip(r.normal(57, 18, n_patients), 18, 96).round().astype(int)
    sex = r.choice(["F", "M", "X"], n_patients, p=[0.49, 0.49, 0.02])
    bmi = np.clip(r.normal(27.5, 5.5, n_patients), 15, 55).round(1)
    smoker = (r.random(n_patients) < 0.18 + 0.002 * (age - 40)).astype(int)
    diabetes = (r.random(n_patients) < 1 / (1 + np.exp(-(-3.2 + 0.045 * (age - 50) + 0.09 * (bmi - 25))))).astype(int)
    hypertension = (r.random(n_patients) < 1 / (1 + np.exp(-(-2.4 + 0.05 * (age - 50) + 0.3 * smoker + 0.12 * (bmi - 25))))).astype(int)
    condition = np.array(
        [
            "diabetes" if diabetes[i] else ("hypertension" if hypertension[i] else "healthy")
            for i in range(n_patients)
        ]
    )
    # add cardiac/neuro/respiratory with age- and risk-dependent probabilities
    condition = np.where(r.random(n_patients) < 0.06 + 0.0015 * (age - 50), "cardiac", condition)
    condition = np.where(r.random(n_patients) < 0.035 + 0.0008 * (age - 50), "neuro", condition)
    condition = np.where((r.random(n_patients) < 0.04 + 0.006 * smoker), "respiratory", condition)
    region = r.choice(["urban", "suburban", "rural"], n_patients, p=[0.55, 0.3, 0.15])
    patients = pd.DataFrame(
        {
            "patient_id": pids,
            "full_name": [synthetic_name(r) for _ in range(n_patients)],
            "age": age,
            "sex": sex,
            "bmi": add_missing(pd.Series(bmi), r, 0.04, "MAR", pd.Series(age)),
            "smoker": smoker,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "primary_condition": condition,
            "region": region,
            "insurance": r.choice(["private", "public", "none"], n_patients, p=[0.55, 0.38, 0.07]),
            "synthetic_source": "seeded_healthcare_simulator",
        }
    )

    enc_pid = r.choice(pids, n_encounters, replace=True)
    pmap = patients.set_index("patient_id")
    admit = random_dates(r, n_encounters, "2022-01-01", "2025-06-30")
    dept = r.choice(["cardiology", "neurology", "internal", "pulmonology", "emergency"], n_encounters, p=[0.2, 0.12, 0.35, 0.13, 0.2])
    los = np.clip(r.gamma(2.2, 1.3, n_encounters) + (dept == "emergency") * r.uniform(0.5, 3, n_encounters), 0.5, 30).round(1)
    discharge = admit + pd.to_timedelta(los * 24, unit="h")
    cost = (los * r.uniform(700, 2400, n_encounters) + r.lognormal(7.2, 0.6, n_encounters)).round(2)
    enc_ids = [f"E{i:06d}" for i in range(1, n_encounters + 1)]
    readmit = (r.random(n_encounters) < 0.11 + 0.1 * (los > 7) + 0.05 * (pmap.loc[enc_pid, "primary_condition"].to_numpy() == "cardiac")).astype(int)
    encounters = pd.DataFrame(
        {
            "encounter_id": enc_ids,
            "patient_id": enc_pid,
            "admit_date": admit,
            "discharge_date": discharge,
            "department": dept,
            "length_of_stay_days": los,
            "cost_usd": cost,
            "readmitted_30d": readmit,
        }
    )

    # diagnoses
    d_rows = []
    diag_id = 1
    for _i, row in encounters.iterrows():
        cond = pmap.loc[row.patient_id, "primary_condition"]
        codes = [ICD.get(cond, ("Z00.0", "Routine exam (synthetic label)"))]
        if cond != "healthy" and r.random() < 0.4:
            extra = r.choice([c for c in ICD if c != cond])
            codes.append(ICD[extra])
        if r.random() < 0.12:
            codes.append(("J18.9", "Pneumonia (synthetic label)"))
        for code, desc in codes:
            d_rows.append(
                {
                    "diagnosis_id": f"D{diag_id:07d}",
                    "encounter_id": row.encounter_id,
                    "icd10_code": code,
                    "description": desc,
                    "is_primary": int(code == codes[0][0]),
                }
            )
            diag_id += 1
    diagnoses = pd.DataFrame(d_rows)

    # labs: correlated with condition, with missingness and implausible sentinels
    lab_rows = []
    lab_id = 1
    for _i, row in encounters.iterrows():
        cond = pmap.loc[row.patient_id, "primary_condition"]
        # creatinine
        base_cr = np.exp(r.normal(np.log(0.95), 0.22))
        if cond in ("diabetes", "hypertension", "cardiac"):
            base_cr *= r.uniform(1.05, 1.5)
        if r.random() < 0.18:
            base_cr = -1.0  # sentinel to be detected and flagged on ingest
        lab_rows.append({"lab_id": f"L{lab_id:08d}", "encounter_id": row.encounter_id, "test": "creatinine", "value": round(float(base_cr), 3), "unit": "mg/dL"})
        lab_id += 1
        hba1c = 5.2 + (4.2 if cond == "diabetes" else 0) + r.normal(0, 0.5)
        if r.random() < 0.25:
            hba1c = np.nan
        lab_rows.append({"lab_id": f"L{lab_id:08d}", "encounter_id": row.encounter_id, "test": "hba1c", "value": round(float(hba1c), 2), "unit": "%"})
        lab_id += 1
        trop = np.exp(r.normal(np.log(0.02), 0.5))
        if cond == "cardiac":
            trop *= r.uniform(3, 20)
        lab_rows.append({"lab_id": f"L{lab_id:08d}", "encounter_id": row.encounter_id, "test": "troponin", "value": round(float(trop), 4), "unit": "ng/mL"})
        lab_id += 1
        # occasional missing row (MAR): entire lab absent for older patients
    labs = pd.DataFrame(lab_rows)
    if len(labs):
        drop_mask = r.random(len(labs)) < 0.03
        labs = labs.loc[~drop_mask].copy()

    med_rows = []
    med_id = 1
    for _i, row in encounters.iterrows():
        cond = pmap.loc[row.patient_id, "primary_condition"]
        drugs = DRUGS.get(cond, [])
        if drugs and r.random() < 0.85:
            for drug in r.choice(drugs, size=int(min(len(drugs), r.integers(1, 3))), replace=False):
                med_rows.append(
                    {
                        "medication_id": f"M{med_id:08d}",
                        "encounter_id": row.encounter_id,
                        "drug": str(drug),
                        "dose_mg": float(np.round(r.choice([2.5, 5, 10, 20, 40, 80, 500, 850]), 2)),
                        "frequency": str(r.choice(["daily", "twice_daily", "as_needed"], p=[0.6, 0.3, 0.1])),
                    }
                )
                med_id += 1
    medications = pd.DataFrame(med_rows)

    # ECG / MRI metadata linked to patients; generation happens in the media simulators.
    ecg_class = np.where(
        condition == "cardiac",
        r.choice(["afib_like", "st_elevation_like", "normal"], n_patients, p=[0.45, 0.3, 0.25]),
        np.where(
            condition == "hypertension",
            r.choice(["normal", "brady", "tachy", "pvc_like"], n_patients, p=[0.55, 0.15, 0.2, 0.1]),
            r.choice(["normal", "brady", "tachy", "afib_like"], n_patients, p=[0.78, 0.08, 0.09, 0.05]),
        ),
    )
    ecg = pd.DataFrame(
        {
            "ecg_id": [f"ECG{i:06d}" for i in range(1, n_patients + 1)],
            "patient_id": pids,
            "class": ecg_class,
            "recorded_at": random_dates(r, n_patients, "2023-01-01", "2025-06-01"),
        }
    )
    mri = pd.DataFrame(
        {
            "study_id": [f"MRI{i:06d}" for i in range(1, n_patients + 1)],
            "patient_id": pids,
            "contrast": r.choice(["T1", "T2", "FLAIR"], n_patients, p=[0.4, 0.3, 0.3]),
            "lesion": ((condition == "neuro") | (r.random(n_patients) < 0.12 + 0.001 * (age - 50))).astype(int),
            "study_date": random_dates(r, n_patients, "2023-01-01", "2025-06-01"),
        }
    )
    return {
        "patients": patients,
        "encounters": encounters,
        "diagnoses": diagnoses,
        "labs": labs,
        "medications": medications,
        "ecg_records": ecg,
        "mri_studies": mri,
    }


RELATIONSHIPS = {
    "tables": {
        "patients": {"primary_key": ["patient_id"]},
        "encounters": {
            "primary_key": ["encounter_id"],
            "foreign_keys": [{"columns": ["patient_id"], "ref_table": "patients", "ref_columns": ["patient_id"]}],
        },
        "diagnoses": {
            "primary_key": ["diagnosis_id"],
            "foreign_keys": [{"columns": ["encounter_id"], "ref_table": "encounters", "ref_columns": ["encounter_id"]}],
        },
        "labs": {
            "primary_key": ["lab_id"],
            "foreign_keys": [{"columns": ["encounter_id"], "ref_table": "encounters", "ref_columns": ["encounter_id"]}],
        },
        "medications": {
            "primary_key": ["medication_id"],
            "foreign_keys": [{"columns": ["encounter_id"], "ref_table": "encounters", "ref_columns": ["encounter_id"]}],
        },
        "ecg_records": {
            "primary_key": ["ecg_id"],
            "foreign_keys": [{"columns": ["patient_id"], "ref_table": "patients", "ref_columns": ["patient_id"]}],
        },
        "mri_studies": {
            "primary_key": ["study_id"],
            "foreign_keys": [{"columns": ["patient_id"], "ref_table": "patients", "ref_columns": ["patient_id"]}],
        },
    }
}

DDL = {
    "patients": "patient_id TEXT PRIMARY KEY, full_name TEXT, age INTEGER, sex TEXT, bmi REAL, smoker INTEGER, diabetes INTEGER, hypertension INTEGER, primary_condition TEXT, region TEXT, insurance TEXT, synthetic_source TEXT",
    "encounters": "encounter_id TEXT PRIMARY KEY, patient_id TEXT REFERENCES patients(patient_id), admit_date TEXT, discharge_date TEXT, department TEXT, length_of_stay_days REAL, cost_usd REAL, readmitted_30d INTEGER",
    "diagnoses": "diagnosis_id TEXT PRIMARY KEY, encounter_id TEXT REFERENCES encounters(encounter_id), icd10_code TEXT, description TEXT, is_primary INTEGER",
    "labs": "lab_id TEXT PRIMARY KEY, encounter_id TEXT REFERENCES encounters(encounter_id), test TEXT, value REAL, unit TEXT",
    "medications": "medication_id TEXT PRIMARY KEY, encounter_id TEXT REFERENCES encounters(encounter_id), drug TEXT, dose_mg REAL, frequency TEXT",
    "ecg_records": "ecg_id TEXT PRIMARY KEY, patient_id TEXT REFERENCES patients(patient_id), class TEXT, recorded_at TEXT",
    "mri_studies": "study_id TEXT PRIMARY KEY, patient_id TEXT REFERENCES patients(patient_id), contrast TEXT, lesion INTEGER, study_date TEXT",
}


def write_healthcare(root: str | Path, seed: int = 13, **sizes: int) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tables = simulate_healthcare(seed=seed, **sizes)
    db = root / "healthcare.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        for name, ddl in DDL.items():
            conn.execute(f"CREATE TABLE {name} ({ddl})")
        for name, df in tables.items():
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
                elif out[c].dtype == bool:
                    out[c] = out[c].astype(int)
            out.where(pd.notna(out), None).to_sql(name, conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()
    (root / "relationships.json").write_text(json.dumps(RELATIONSHIPS, indent=2), encoding="utf-8")
    csv_dir = root / "healthcare_csv"
    csv_dir.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(csv_dir / f"{name}.csv", index=False)
    (csv_dir / "relationships.json").write_text(json.dumps(RELATIONSHIPS, indent=2), encoding="utf-8")
    return db
