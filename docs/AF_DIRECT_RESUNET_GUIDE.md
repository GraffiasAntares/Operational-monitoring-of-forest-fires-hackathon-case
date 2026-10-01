# AF Direct ResUNet: запуск эксперимента

Этот эксперимент заменяет неудачную residual-схему `add/remove` на прямую сегментацию fire-маски. Текущий submission-пайплайн не меняется.

## Что реализовано

- `AfDirectResUNet`: residual U-Net, 26 thermal-контекстных входов, один логит fire на пиксель;
- base width 24: **4 569 505 параметров**;
- строгий пространственно-временной train/validation split;
- обучающие патчи 256x256: 50% fire, 25% горячие ложные объекты `I4 > 310`, 25% обычный фон;
- геометрические аугментации без искажения температурных каналов;
- loss = 0.5 weighted BCE + 0.5 focal Tversky;
- validation по полным чипам, с подбором порога вероятности по F1.

## 1. Подготовка данных

Подготовку уже можно выполнить без GPU и без обучения:

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe scripts\prepare_af_direct_data.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --output-dir ml\artifacts\af_direct_strict_v1
```

Ожидаемый строгий split: примерно 216 train, 46 validation и 158 excluded. Скрипт не копирует растры: создаёт manifest и нормализацию только по train.

## 2. Обучение на RTX 2050

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe scripts\train_af_direct.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --prepared-dir ml\artifacts\af_direct_strict_v1 `
  --output-dir ml\artifacts\af_direct_strict_v1\checkpoints `
  --patch-size 256 --patches-per-epoch 512 `
  --batch-size 4 --accumulation 2 `
  --base-channels 24 `
  --epochs 70 --patience 12 --lr 2e-4 --seed 42
```

Если CUDA out of memory:

```text
--batch-size 2 --accumulation 4
```

Не меняй одновременно sampler, loss и ширину модели: иначе нельзя будет понять причину изменения метрики.

## 3. Проверка лучшего checkpoint

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe scripts\evaluate_af_direct.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --prepared-dir ml\artifacts\af_direct_strict_v1 `
  --checkpoint ml\artifacts\af_direct_strict_v1\checkpoints\best.pt
```

Скрипт выведет и запишет F1, IoU, precision, recall и лучший threshold в `strict_validation_metrics.json` рядом с checkpoint.

## Критерий решения

Нижняя планка — thermal baseline `I4 > 330`: strict-validation F1 **0.8161**. Пока модель не обходит её с запасом хотя бы 0.01, в общий submission она не попадает.
