# Данные, метрики и submission

## Структура задач

AF и BS разделены не только в датасете, но и в исходном коде:

```text
ml/
  af/  # Active Fire: VIIRS, бинарная маска, F1, split по спутниковому пролёту
  bs/  # Burn Severity: Sentinel-1/2, классы 0-3, IoU/mIoU, split по fire_event_id
```

Новый код должен импортировать нужную задачу напрямую: `ml.af` или `ml.bs`.
Файлы `ml/data.py` и `ml/metrics.py` сохранены только как совместимый слой для
уже существующих вызовов; они не объединяют обучение двух задач.

## Обучающие данные

Корень датасета передаётся как каталог, содержащий `train/af` и `train/bs`.

- AF: `viirs` (I1-I5, углы и valid), `aux` (покров, DEM, метео) и бинарная
  `masks`. Загрузчик возвращает 13 каналов.
- BS: Sentinel-2 до/после, Sentinel-1 до/после, `aux` и маска 0-3.
  Загрузчик возвращает 27 каналов.

`FireChipDataset` проверяет наличие всех слоёв для каждого ID и их
геопривязку. Пиксели с target `255` исключаются из loss и метрик. Облачные
пиксели BS не исключаются: они остаются частью задания.

```python
from ml.bs import BurnSeverityDataset, grouped_split

dataset = BurnSeverityDataset(r"D:\ForestFire\fire-train-renamed")
sample = dataset.load(0)
split = grouped_split(dataset.records, seed=42)
```

Готовый файл split можно получить без кода модели:

```powershell
python scripts/build_ml_split.py --data-dir D:\ForestFire\fire-train-renamed --kind bs --output splits/bs.json
```

BS делится только группами `fire_event_id`. Для AF идентификатор пожара
намеренно не выдан, поэтому splitter сохраняет целый спутниковый пролёт в
одной части выборки.

### Предобработка BS

`BurnSeverityPreprocessor` переводит Sentinel-2 из шкалы `0..10000` в
отражение, Sentinel-1 из `dB * 100` в dB, рассчитывает NBR, dNBR и RdNBR,
а также разности оптических и SAR-каналов между датами. SCL и ESA WorldCover
не подаются как числа: каждый из них превращается в one-hot признаки. Это
сохраняет информацию об облаках и тенях, не исключая их из обучения или
инференса. На выходе 74 канала.

Статистики стандартизации необходимо обучать **только на train-части**
группового сплита и сохранять рядом с весами:

```python
from ml.bs import BurnSeverityDataset, BurnSeverityPreprocessor, grouped_split

dataset = BurnSeverityDataset(r"D:\ForestFire\fire-train-renamed")
split = grouped_split(dataset.records, seed=42)
train_indices = [i for i, record in enumerate(dataset.records) if record.chip_id in split["train"]]

preprocessor = BurnSeverityPreprocessor().fit(dataset.load_layers(i) for i in train_indices)
preprocessor.statistics.save("artifacts/bs_preprocessing.npz")
sample = dataset.load(train_indices[0], preprocessor=preprocessor)
```

Для экспериментов с U-Net доступны режимы `s2`, `s2_aux`, `s2_s1_aux` и
`full`; они соответствуют абляциям B-E. Таблица результатов строится одной
командой после baseline и запусков обучения:

```powershell
python scripts/summarize_bs_ablations.py --baseline ml/artifacts/dnbr_baseline_seed42/metrics.json --run B=ml/artifacts/bs_s2/history.json --run C=ml/artifacts/bs_s2_aux/history.json --run D=ml/artifacts/bs_s2_s1_aux/history.json --run E=ml/artifacts/bs_full/history.json --output ml/artifacts/bs_ablations.csv
```

Для повторяемого запуска лучше сначала сохранить сплит, затем передать его
утилите:

```powershell
python scripts/build_ml_split.py --data-dir D:\ForestFire\fire-train-renamed --kind bs --output splits/bs.json
python scripts/fit_bs_preprocessing.py --data-dir D:\ForestFire\fire-train-renamed --split splits/bs.json --output ml/artifacts/bs_preprocessing.npz
```

## Метрика

`ml.metrics.competition_score` вычисляет конкурсную формулу:

```text
0.35 * F1_af + 0.35 * IoU_burn + 0.30 * mIoU_sev
```

Все показатели микро-усреднены по пикселям. Для классов тяжести 1-3, которых
нет одновременно в эталоне и предсказании, IoU равен 1 — это правило кейса.

## Submission

`ml.submission.write_submission` строит CSV строго в порядке строк
`sample_submission.csv`. RLE идёт построчно, слева направо и сверху вниз,
с нумерацией пикселей от единицы.

```python
from ml.submission import write_submission

write_submission("sample_submission.csv", predictions, "submission.csv")
```

Где `predictions` — словарь `{chip_id: ndarray[height, width]}` со значениями
0-3. Проверить готовый файл можно так:

```powershell
python scripts/validate_submission.py --template sample_submission.csv --submission submission.csv
```

Проверка требует те же `chip_id`/`class_id`, тот же порядок строк, валидные
границы RLE и отсутствие пересечений масок классов BS.
