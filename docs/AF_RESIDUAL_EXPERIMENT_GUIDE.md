# AF: thermal candidate generator + residual U-Net

Этот эксперимент проверяет первые две гипотезы одновременно, не заменяя текущий AF-алгоритм в submission-пайплайне.

## Идея

Тепловое правило `I4 > 330` на строгой валидации уже даёт F1 около **0.8161**. Вместо того чтобы обучать U-Net с нуля искать очень редкие пиксели пожара, используем его как корректор этого физически осмысленного правила.

1. `I4 > 310` — широкая кандидатная область: в неё может попасть пожар, который не прошёл строгий порог.
2. `I4 > 330` — базовая маска с хорошей точностью.
3. U-Net выдаёт две независимые карты:
   - **add**: добавить пиксель из широкой кандидатной области к базовой маске;
   - **remove**: удалить ложное срабатывание из базовой маски.
4. Итоговая маска: `((I4 > 330) OR (I4 > 310 AND add)) AND NOT(remove)`.

Так сеть никогда не «нарисует» пожар в полностью холодной области: она только исправляет ошибки thermal-кандидата. Это уменьшает дисбаланс классов и делает ошибку модели более контролируемой.

## Входы сети

26 каналов: исходные `I1…I5`, геометрия съёмки, valid mask, landcover и метео/рельеф; спектральные признаки `I4-I5`, `I3-I4`, `I3-I5`, `I4/I5`; локальные аномалии I4 для окон 5/11/21/41 и локальная дисперсия; четыре бинарные thermal-маски для порогов I4 300/310/320/330 K.

Нормализация считается **только по обучающей части** строгого пространственно-временного split. Валидационный осенний регион не участвует ни в статистиках, ни в обучении.

## Шаг 1. Подготовить manifest и статистики

Запускать из `D:\projs\ForestFire` в уже проверенном окружении transcribAI:

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe scripts\prepare_af_residual_data.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --output-dir ml\artifacts\af_residual_strict_v1
```

Это не обучение: скрипт читает обучающие чипы, создаёт `manifest.csv` и `normalization.json`. В последнем проверь `chip_counts` (ожидается strict split примерно 216/46/158) и числа `correction_pixels`: обе задачи коррекции должны иметь ненулевые positive-пиксели.

## Шаг 2. Обучить первую модель

RTX 2050 (4 GB) подходит для base width 16, batch 2 и AMP. Effective batch получается 8 за счёт накопления градиента.

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe scripts\train_af_residual.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --prepared-dir ml\artifacts\af_residual_strict_v1 `
  --output-dir ml\artifacts\af_residual_strict_v1\checkpoints `
  --epochs 40 --patience 10 `
  --batch-size 2 --accumulation 4 `
  --base-channels 16 --lr 2e-4 --seed 42 `
  --add-threshold 0.5 --remove-threshold 0.5
```

Если CUDA out of memory, не меняй архитектуру сразу: используй `--batch-size 1 --accumulation 8`. Если обучение стабильно помещается и есть запас VRAM, следующим отдельным опытом можно проверить `--base-channels 20`; сравнивать нужно на том же strict split.

Скрипт использует геометрические аугментации, безопасные для растров: случайные horizontal/vertical flip и поворот на 0/90/180/270 градусов. Яркостные аугментации не применяются: они разрушили бы физический смысл порогов температуры I4.

## Что смотреть в логе

`validation_f1` — F1 уже после сборки итоговой маски из baseline, add и remove, а не F1 отдельной головы. Начальная практическая цель — превзойти thermal baseline **0.8161** на strict validation хотя бы на 0.01. Результат около или ниже baseline означает, что корректор пока только вносит шум; тогда submission оставляем на текущем thermal AF.

Лучший checkpoint и журнал эпох появятся здесь:

```text
ml\artifacts\af_residual_strict_v1\checkpoints\best.pt
ml\artifacts\af_residual_strict_v1\checkpoints\history.json
```

## Шаг 3. Калибровать пороги без переобучения

После обучения один раз прогоняем checkpoint на validation и перебираем сетку add/remove-порогов. Это только inference: веса и checkpoint не меняются.

```powershell
D:\university\bdiploma\venv311\Scripts\python.exe scripts\evaluate_af_residual.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --prepared-dir ml\artifacts\af_residual_strict_v1 `
  --checkpoint ml\artifacts\af_residual_strict_v1\checkpoints\best.pt `
  --batch-size 2
```

Лучшие десять комбинаций выводятся в консоль, все результаты сохраняются в `calibration.json` рядом с checkpoint. В этом же файле будет F1 чистого `I4 > 330` на тех же валидных пикселях.

## Первые контролируемые абляции (после первого запуска)

Менять только один фактор за раз и сохранять новый output-dir.

1. **Пороги fusion:** использовать лучший результат `calibration.json`; для этого не нужно обучать модель повторно. Более высокий add-порог обычно уменьшает ложные срабатывания.
2. **Ширина thermal-кандидата:** сначала оставить 310 K. Если модель улучшает precision, но не recall, следующим кодовым опытом проверить 300 K; если добавляет слишком много ложных пикселей — 315 или 320 K.
3. **Размер модели:** только если base=16 обошёл baseline и validation F1 ещё растёт к окончанию обучения, проверить base=20. Большая сеть не должна быть первым рычагом: данных мало, а задача — локальная коррекция простой эвристики.

## Важное ограничение

Пока этот опыт изолирован: код подготовки и обучения добавлен, но production AF inference и submission не менялись. Внедрять checkpoint в общий пайплайн стоит только после честного улучшения strict validation и отдельной проверки на отложенных тестовых чипах.
