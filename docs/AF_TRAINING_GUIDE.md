# AF: подготовка данных и обучение U-Net на RTX 2050

Этот документ описывает первый воспроизводимый эксперимент для модуля
активного горения. Его задача — получить стабильный validation F1, а не
максимально большую сеть.

## Что уже подготовлено

Скрипт подготовки не копирует исходные GeoTIFF и не меняет датасет. Он создаёт
малые train-derived артефакты:

- `manifest.csv` — детерминированный split по спутниковым пролётам;
- `normalization.json` — mean/std, посчитанные только по train-чипам;
- перечень из 15 входных признаков и доля позитивных пикселей.

```powershell
python scripts/prepare_af_data.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --output-dir ml\artifacts\af_v1 `
  --seed 42
```

Для честной оценки private-like обобщения вместо обычного split используйте
строгий spatial-and-seasonal holdout:

```powershell
python scripts/prepare_af_data.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --output-dir ml\artifacts\af_strict_v1 `
  --seed 42 `
  --strict
```

Ожидаемый результат для текущего набора: 333 train-чипа, 87 validation-чипов,
а доля огненных пикселей в обеих частях около 0.0355%.

## Входные признаки

Базовый вектор имеет 15 каналов:

```text
I1, I2, I3, I4, I5, solar_zenith, sensor_zenith,
valid, landcover, DEM, t2m, rh2m, wind_speed,
I4_minus_I5, I4_local_anomaly_21
```

`valid` остаётся бинарным, `landcover` — категориальным. Непрерывные признаки
нормализуются train-статистиками. Отсутствующие непрерывные значения
заполняются train mean, а `target == 255` исключается из loss и метрик.

Производные thermal-признаки — гипотеза, а не данность. Обязательно сравнить
модель только с исходными 13 каналами и модель с 15 каналами.

## Настройка среды Windows

1. Убедиться, что драйвер NVIDIA видит GPU:

   ```powershell
   nvidia-smi
   ```

2. Создать окружение в корне репозитория. Python 3.11 или 3.12 подходит:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

3. Установить CUDA-сборку PyTorch. Выбрать Windows / Pip / Stable и подходящую
   CUDA-сборку в официальном селекторе PyTorch. Для RTX 2050 обычно разумно
   начать с CUDA 11.8, если драйвер её поддерживает:

   ```powershell
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. Проверить GPU из Python:

   ```powershell
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
   ```

Если последний вызов выводит `False`, не начинать долгий запуск: сначала
переустановить PyTorch по официальному селектору или обновить драйвер.

## Конфигурация первого запуска

```text
model:              U-Net, 15 input channels, 1 output logit
base_channels:      16
downsampling:       4 уровня
normalization:      GroupNorm
activation:         SiLU
dropout:            0.05
image size:         256 × 256, без crop
batch size:         4
precision:          mixed precision (torch.autocast + GradScaler)
gradient accumulation: 2
effective batch:    8
optimizer:          AdamW(lr=3e-4, weight_decay=1e-4)
scheduler:          cosine decay
epochs:             до 50
early stopping:     patience=10 по validation F1
seed:               42
```

Если CUDA выдаёт out-of-memory, последовательно снизить `batch_size` до 2,
затем до 1; `base_channels=16` не менять в первом эксперименте. Gradient
accumulation сохранить так, чтобы эффективный batch был не меньше 4.

## Первый запуск

После подготовки данных выполнить из активированного `venv311`:

```powershell
python scripts\train_af.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --prepared-dir ml\artifacts\af_v1 `
  --output-dir ml\artifacts\af_v1\checkpoints `
  --epochs 50 `
  --batch-size 4 `
  --accumulation 2 `
  --workers 0
```

На Windows оставить `--workers 0` для первого запуска: так проще исключить
ошибки дочерних процессов и конфликтов GDAL/rasterio. Если GPU сообщает
`out of memory`, повторить с `--batch-size 2 --accumulation 4`.

После завершения в `output-dir` появятся:

- `best.pt` — лучший checkpoint и подобранный порог;
- `history.json` — loss, F1 и threshold всех эпох.

## Sampling и loss

Пустые чипы нельзя удалять: они критичны для контроля ложных тревог. Но в
каждом train epoch нужно делать примерно 70% выборок из чипов, где есть огонь,
и 30% — из пустых. Validation всегда остаётся в исходном распределении.

Рекомендуемый первый loss:

```text
loss = 0.7 × focal_bce(logits, target, alpha=0.75, gamma=2)
     + 0.3 × dice_loss(sigmoid(logits), target)
```

Loss считается только на valid-пикселях. Нельзя использовать accuracy как
основную метрику: предсказание «везде 0» будет почти идеальным по accuracy, но
получит нулевой F1.

## Обязательные эксперименты

Все эксперименты должны использовать одинаковые split, seed, sampler и loss.

| Код | Вход | Цель |
| --- | --- | --- |
| A | 13 исходных каналов | исходный baseline U-Net |
| B | 13 + `I4-I5` | проверить физически осмысленную разность |
| C | 13 + обе thermal-фичи | проверить локальный контраст |
| D | лучший вход, `base_channels=24` | проверить, нужна ли большая модель |

Для каждого checkpoint подбирать порог sigmoid только на validation, например
по сетке от `0.05` до `0.80` с шагом `0.01`. Сохранять threshold, давший лучший
микро-F1. Значение `0.5` не считать значением по умолчанию.

## Критерий выбора модели

Сохранять checkpoint с максимальным validation F1. После выбора лучшего
эксперимента повторить его минимум на двух дополнительных seed. Увеличение
размера модели принимать только если прирост F1 больше 0.01 и повторяется;
иначе оставлять `base_channels=16`.

## Что зафиксировать после обучения

- `normalization.json`, split и seed;
- конфигурацию модели;
- лучшую эпоху, threshold и validation F1;
- confusion matrix: TP, FP, FN;
- примеры TP, FP и FN: особенно техногенные точки, блики, горячий грунт и
  края облаков;
- веса модели и точную команду инференса.

Не использовать для предсказаний тестовых чипов внешние готовые продукты
активного огня или попытки восстановить их время/место: это запрещено
правилами соревнования.
