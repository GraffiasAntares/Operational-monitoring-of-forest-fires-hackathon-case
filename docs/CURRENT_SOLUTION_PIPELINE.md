# Текущий pipeline решения AF + BS

Документ описывает фактически собранный путь от официального test-архива до
готового `submission.csv`. Он предназначен как техническая точка отсчёта перед
экспериментами: менять компоненты можно, но сравнивать их следует на одинаковых
пространственно-временных holdout, а не на скрытом test.

## 1. Общая схема

```text
D:\fire-test-renamed\test
 ├─ AF: VIIRS + aux ──> I4 > 330 K ───────> маска 256×256 класса 1
 └─ BS: S2/S1/aux ────> preprocessing ────> U-Net ──> маска 512×512 классов 0–3
                                                           │
                         sample_submission.csv <──────────┘
                                           │
                                           └──> submission.csv (RLE)
```

Точка входа — [inference.py](../inference.py). Он читает общий
`test/meta.csv`, различает AF и BS по полю `kind`, формирует все маски и пишет
строки строго в порядке `test/sample_submission.csv`.

В текущем test-наборе:

| Задача | Чипов | Размер маски | Строк на чип в submission |
| --- | ---: | --- | ---: |
| AF | 180 | 256×256 | 1 (class 1) |
| BS | 89 | 512×512 | 3 (classes 1, 2, 3) |
| Итого | 269 | — | 447 |

Класс 0 нигде в CSV не записывается.

## 2. Решение AF — Active Fire

### Вход

Используется файл `test/af/viirs/<chip>_VIIRS_I1-I5.tif`:

- band 4 — `I4`, brightness temperature;
- band 8 — supplied mask `valid`.

Aux-слои и остальные спектральные каналы в текущем production-решении AF не
используются.

### Алгоритм

```text
AF_mask = (I4 > 330 K) AND (valid > 0)
```

Результат — бинарная `uint8` маска класса 1. Её кодирование в RLE идёт
row-major, слева направо и сверху вниз, с one-based позициями.

### Почему выбран именно этот путь

На строгой пространственно-временной validation правило дало F1 `0.8161`,
тогда как текущая AF U-Net показала `0.6177`. На всех доступных AF-территориях
правило дало диагностический F1 `0.8512`; эта общая величина не используется
для выбора модели, поскольку включает train-чипы.

Следовательно, AF сейчас не использует ML по сознательному решению: простой
физический baseline оказался сильнее имеющейся нейросети.

### Артефакты и диагностика

На test rule пометил 61 798 пикселей в 108 из 180 AF-чипов. Это не метрика
качества: test mask скрыта. Промежуточные AF-маски можно получить отдельно
через `scripts/predict_af_thermal_baseline.py`, но единый pipeline генерирует
их в памяти и сразу добавляет в общий CSV.

## 3. Решение BS — Burn Severity

### Входы и feature engineering

U-Net получает 74 канала. Непрерывные признаки стандартизируются по mean/std,
полученным только из 179 train-чипов event-level split (`seed=42`), затем
добавляются one-hot категориальные признаки.

| Группа | Каналов | Содержание |
| --- | ---: | --- |
| Sentinel-2 | 18 | 9 отражательных каналов до и после; значения `/10000` |
| Sentinel-1 | 4 | VV/VH до и после; значения `/100` dB |
| Рельеф | 2 | DEM, slope |
| Индексы пожара | 4 | NBR pre/post, dNBR, RdNBR |
| Изменения | 11 | 9 разностей S2 и 2 разности SAR |
| SCL | 24 | one-hot 12 классов до и после |
| Land cover | 11 | one-hot ESA WorldCover |
| **Итого** | **74** | — |

Ключевая формула:

```text
NBR  = (B8A - B12) / (B8A + B12)
dNBR = NBR_pre - NBR_post
```

Облачные пиксели не выкидываются: конкурс их оценивает. Их состояние передаётся
через SCL, а Sentinel-1 остаётся доступным источником сигнала.

### Модель

Используется checkpoint:

```text
ml/artifacts/bs_unet_full_tuned_seed42/best.pt
```

Это компактная четырёхуровневая U-Net с GroupNorm и SiLU:

| Параметр | Значение |
| --- | --- |
| Входов | 74 |
| Базовых каналов | 24 |
| Dropout | 0.10 |
| Выход | 4 logits на пиксель, затем `argmax` |
| Обучение checkpoint | batch 1, accumulation 4, class weights `[1, 4, 4, 4]`, seed 42 |
| Schedule | 2 эпохи warm-up, затем cosine decay `3e-4 -> 1e-6` |
| Loss / regularization | weighted CE + Dice (вес Dice 0.60), weight decay `3e-5` |
| Лучшая сохранённая эпоха | 63 |

Checkpoint обучался с event-level split: чипы одного `fire_event_id` не
смешиваются между train и validation. В его сохранённых validation-метриках:

| Метрика BS | Значение |
| --- | ---: |
| IoU burn | 0.6244 |
| IoU severity 1 | 0.4085 |
| IoU severity 2 | 0.5962 |
| IoU severity 3 | 0.6959 |
| mIoU severity | 0.5669 |
| `0.35×IoU_burn + 0.30×mIoU` | 0.3886 |

Эти значения — локальная validation checkpoint, не результат скрытого test.

### Post-training logit calibration

Перед `argmax` общий pipeline применяет фиксированные additive bias к logits:

```text
[class 0, class 1, class 2, class 3] = [0.00, -0.45, -0.15, +0.60]
```

Bias подобраны скриптом `scripts/calibrate_bs_logits.py` на тех же 45
event-level validation-чипах, которые не входили в обучение checkpoint. Это не
изменяет веса U-Net и не является обработкой готовой дискретной маски: bias
добавляется к четырём logits, затем выполняется `argmax`.

| Вариант на validation | IoU burn | mIoU severity | BS score |
| --- | ---: | ---: | ---: |
| Обычный `argmax` | 0.63376 | 0.57187 | 0.39338 |
| Bias-калибровка | **0.63795** | **0.57331** | **0.39528** |

Прирост малый (`+0.00190` BS score) и получен на одном validation split, поэтому
его нужно рассматривать как рабочую гипотезу, а не доказанную универсальную
оптимизацию. Файл `logit_calibration.json` включён в pipeline по умолчанию;
перед серьёзным сравнением следует повторить калибровку на других event folds.

### Confidence-gated U-Net + dNBR hybrid

Для hybrid dNBR-пороги калибруются только на 179 train-чипах и отдельно для
групп landcover. На validation dNBR не заменяет модель целиком. Лучшая схема:

```text
если confidence(U-Net после logit bias) < 0.60
и U-Net выбрала class 0
и dNBR предсказывает class 1/2/3:
    заменить 0 на dNBR-класс
иначе:
    оставить U-Net prediction
```

Это режим `burn_promote`: он повышает recall гари только в сомнительных
background-пикселях. На event-level validation:

| Вариант | IoU burn | mIoU severity | BS score |
| --- | ---: | ---: | ---: |
| U-Net + logit calibration | 0.63795 | **0.57331** | 0.39528 |
| Hybrid U-Net + dNBR | **0.64294** | 0.57026 | **0.39611** |

Hybrid повышает burn recall и итоговый BS score на `+0.00083`, но немного
ухудшает mIoU severity. Поэтому он включён как текущая экспериментальная
конфигурация, а не как доказанно лучший вариант: проверка на дополнительных
event folds обязательна. Конфигурация находится в
`ml/artifacts/bs_unet_full_tuned_seed42/dnbr_hybrid.json` и автоматически
подхватывается `inference.py`.

### Важный preprocessing-артефакт

Checkpoint требует файл:

```text
ml/artifacts/bs_preprocessing_seed42.npz
```

Он поставляется с tuned checkpoint в Git. При необходимости его можно
детерминированно пересоздать командами из раздела 5. Нельзя заменять его
статистикой, подсчитанной на test: это изменит входное распределение модели и
будет некорректно.

## 4. Сборка и проверка submission

`inference.py` получает:

1. AF dictionary: `chip_id -> uint8(256,256)`;
2. BS dictionary: `chip_id -> uint8(512,512)`;
3. объединяет словари;
4. вызывает `ml.submission.write_submission()`.

Функция берёт именно template и для каждой его строки кодирует нужный класс в
RLE. Это гарантирует порядок и наличие всех AF/BS строк. После этого
`scripts/validate_submission.py` проверяет header, порядок строк, RLE-границы,
формы масок и отсутствие пересечений классов BS.

Созданный текущий файл
`ml/artifacts/submission/submission.csv` прошёл локальный валидатор и был
принят сервером с результатом **0.67**. Это фактический внешний результат,
но не повод подбирать параметры по test после отправки.

## 5. Воспроизводимый запуск

В чистой копии сначала один раз подготовить BS statistics:

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe scripts\build_ml_split.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --kind bs --output ml\artifacts\bs_split_seed42.json --seed 42

D:\university\bdiploma\venv311\Scripts\python.exe scripts\fit_bs_preprocessing.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --split ml\artifacts\bs_split_seed42.json `
  --output ml\artifacts\bs_preprocessing_seed42.npz
```

Полный inference:

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe inference.py `
  --data-dir D:\fire-test-renamed `
  --output ml\artifacts\submission\submission.csv `
  --masks-output ml\artifacts\submission\predicted_masks.npz `
  --device cuda

D:\university\bdiploma\venv311\Scripts\python.exe scripts\validate_submission.py `
  --template D:\fire-test-renamed\test\sample_submission.csv `
  --submission ml\artifacts\submission\submission.csv
```

## 6. Ограничения текущего решения

- AF-порог фиксирован глобально; он не умеет отдельно обрабатывать слабое
  горение и отдельные типы поверхности.
- AF U-Net пока проигрывает порогу, поэтому в production не участвует.
- BS dNBR-only baseline слабее U-Net; он полезен как диагностический слой, но
  сейчас не входит в финальное решение.
- Один validation split не даёт полной оценки переноса между сезонами и
  территориями. Для принятия изменений нужны несколько split по событиям/регионам.
- Файл BS preprocessing следует версионировать вместе с моделью в release
  (например, в отдельном model-artifacts архиве), иначе чистое окружение требует
  его воспроизведения.

## 7. Гипотезы улучшения метрик

Ниже — гипотезы, а не уже подтверждённые улучшения. Каждую проверять без
использования hidden test и фиксировать в журнале экспериментов.

### AF

1. **Гибрид threshold + ML-correction.** Оставить `I4>330` базовой маской,
   добавить её как вход U-Net/малой модели и обучать исправлять только FN слабых
   пожаров и FP горячего фона. Сравнивать с чистым порогом на тех же holdout.
2. **Калибровка порога по валидным регионам.** Проверить фиксированные пороги
   около 320–340 K на нескольких spatial-temporal split. Не выбирать порог по
   hidden test и не использовать тестовые координаты.
3. **Осмысленные признаки контекста.** I4-I5, локальная аномалия I4,
   landcover и метео могут быть полезны именно для correction model, а не как
   попытка заменить сильное физическое правило.
4. **Постобработка компонент.** Проверить удаление одиночных FP-компонент и
   заполнение малых пробелов, но только если это даёт прирост на независимых
   split: огонь может быть реально очень маленьким.

### BS

1. **Ensemble U-Net + dNBR.** dNBR уже находится среди входов. Можно проверить
   rule-based fallback или смешивание логитов для облачных/неуверенных пикселей;
   ключевой критерий — прирост IoU класса 1 без провала class 2/3.
2. **Loss для классов тяжести.** Текущая модель использует class weights. Стоит
   сравнить weighted CE с CE+multiclass Dice, focal loss и Tversky-подходом;
   выбирать по `0.35×IoU_burn + 0.30×mIoU`, а не только по loss.
3. **Кросс-валидация по fire_event_id.** Повторить обучение/оценку хотя бы на
   трёх seed или folds. Это покажет, является ли удачная validation эпохи 20
   устойчивой или особенностью одного набора событий.
4. **Анализ SCL и SAR.** Отдельно измерить качество под облаком и без облака.
   Если под облаком U-Net слабее, усилить роль `ΔVV`, `ΔVH` и SCL, а не
   исключать такие пиксели.
5. **Калибровка классов после U-Net.** Проверить confusion matrix и небольшую
   calibration logits/порогов между классами 1–2–3 на validation. Простая
   морфология допустима лишь после честной проверки.
6. **Ансамбль независимых seed.** Усреднять logits 2–3 U-Net, а затем делать
   `argmax`; на маленьком датасете это часто снижает дисперсию, но требует
   подтверждения на event-level folds.

## 8. Правило для будущих сравнений

Новый вариант можно считать улучшением только если одновременно выполнены три
условия:

1. он использует лишь разрешённые входные данные и не восстанавливает test
   географию/дату;
2. превосходит текущий baseline на заранее выбранных независимых holdout;
3. сохраняет воспроизводимый inference и валидный `submission.csv`.
