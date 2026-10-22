import tempfile, pathlib, traceback
from synthforge.simulator.banking import write_banking
from synthforge.simulator.ecg import write_wfdb_cohort, write_edf_cohort, write_ecg_csv_cohort
from synthforge.simulator.mri import write_mri_nifti, write_mri_dicom
from synthforge.ingest.database import load_database, sqlite_url
from synthforge.ingest.timeseries import load_ecg
from synthforge.ingest.imaging import load_imaging
root = pathlib.Path(tempfile.mkdtemp(prefix='sf_test_'))
print('root', root)
try:
    db = write_banking(root/'bank', seed=7, n_customers=60, n_accounts=90, n_transactions=500, n_loans=80)
    ds = load_database(sqlite_url(db))
    print('bank tables', list(ds.tables), 'rows', ds.n_rows, 'warnings', ds.warnings[:2])
    print('integ', ds.profile.get('relational',{}).get('ok'), ds.profile.get('relational',{}).get('orphan_foreign_keys'))
    w = write_wfdb_cohort(root/'ecg_wfdb', n=4, seed=21)
    e = load_ecg(w)
    print('wfdb', e.signals.signals.shape, e.profile['signals']['classes'], e.profile['signals']['quality_ok_rate'])
    ed = write_edf_cohort(root/'ecg_edf', n=3, seed=22)
    e2 = load_ecg(ed)
    print('edf', e2.signals.signals.shape, e2.signals.lead_names[:3], float(e2.signals.signals.mean()))
    c = write_ecg_csv_cohort(root/'ecg_csv', n=2, seed=23)
    e3 = load_ecg(c)
    print('csv', e3.signals.signals.shape)
    m = write_mri_nifti(root/'mri', n=4, seed=51, resolution=32)
    im = load_imaging(m, target_resolution=32, slices_per_volume=2)
    print('mri', im.images.images.shape, im.images.labels[['study_id','contrast','lesion']].head().to_dict('records'))
    d = write_mri_dicom(root/'dicom', n_studies=1, seed=61, resolution=32)
    im2 = load_imaging(d, target_resolution=32, slices_per_volume=2)
    print('dicom', im2.images.images.shape, im2.images.deid_report['phi_tags_found'])
except Exception:
    traceback.print_exc()
finally:
    print('done')
