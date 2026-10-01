# Единый inference-пайплайн

`inference.py` формирует один валидный `submission.csv` для официального
test-архива. AF по умолчанию решается правилом `I4 > 330 K`, а при передаче
`--af-checkpoint` — direct ResUNet с нормализацией и порогом из checkpoint. BS —
улучшенным чекпойнтом U-Net из `ml/artifacts/bs_unet_full_tuned_seed42/best.pt`.

Файл preprocessing для tuned checkpoint уже находится в репозитории по пути
`ml/artifacts/bs_preprocessing_seed42.npz`. Если его потребуется пересоздать,
он строится детерминированно на seed-42 event split:

```powershell
python scripts\build_ml_split.py --data-dir D:\ForestFire\fire-train-renamed --kind bs --output ml\artifacts\bs_split_seed42.json --seed 42
python scripts\fit_bs_preprocessing.py --data-dir D:\ForestFire\fire-train-renamed --split ml\artifacts\bs_split_seed42.json --output ml\artifacts\bs_preprocessing_seed42.npz
```

Затем запустить полный pipeline:

```powershell
python inference.py --data-dir D:\fire-test-renamed --af-checkpoint ml\artifacts\af_direct_strict_v1\checkpoints\best.pt --output ml\artifacts\submission\submission_direct_af.csv --masks-output ml\artifacts\submission\predicted_masks_direct_af.npz --device cuda
python scripts\validate_submission.py --template D:\fire-test-renamed\test\sample_submission.csv --submission ml\artifacts\submission\submission_direct_af.csv
```

CSV формируется ровно в порядке `sample_submission.csv`: одна RLE-строка класса
1 на AF-чип и три строки классов 1/2/3 на BS-чип. Класс 0 не записывается.

По умолчанию pipeline также применяет
`ml/artifacts/bs_unet_full_tuned_seed42/logit_calibration.json`: это небольшая
калибровка logits BS-модели, проверенная на event-level validation. Чтобы
отключить её для абляции, передайте несуществующий путь, например
`--bs-logit-calibration none.json`.

Если присутствует
`ml/artifacts/bs_unet_full_tuned_seed42/dnbr_hybrid.json`, pipeline также
использует train-calibrated, confidence-gated dNBR hybrid. Отключение для
абляции: `--bs-dnbr-hybrid none.json`.
