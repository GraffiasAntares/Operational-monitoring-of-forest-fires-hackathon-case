# Журнал экспериментов — BS (выгоревшая территория и степень поражения)

Статус: 18 сентября 2026. В этом журнале находятся только проведённые
эксперименты. Метрики не являются оценкой скрытого теста хакатона.

## Метрика BS

Для BS организаторы учитывают `IoU_burn` для объединённой маски классов 1--3 и
`mIoU_sev` — среднее IoU классов 1, 2, 3. В таблицах указан также частичный
вклад BS в итоговый конкурсный score:

```text
BS weighted score = 0.35 * IoU_burn + 0.30 * mIoU_sev
```

Это не полный score кейса: к нему отдельно добавляется результат AF.

## Эксперимент BS-01: dNBR thresholding

Проверена детерминированная физическая эвристика без ML:

```text
NBR_pre  = (B8A_pre  - B12_pre)  / (B8A_pre  + B12_pre)
NBR_post = (B8A_post - B12_post) / (B8A_post + B12_post)
dNBR     = NBR_pre - NBR_post

dNBR < t0       -> class 0, не горело
t0 <= dNBR < t1 -> class 1, слабое поражение
t1 <= dNBR < t2 -> class 2, среднее поражение
dNBR >= t2      -> class 3, сильное поражение
```

### Разбиение

Весь `fire_event_id` помещается только в одну часть разбиения. Использован
детерминированный event-level split, `seed=42`: train — 179 чипов, validation
— 45 чипов. Пороги подбирались исключительно по train-пикселям; результаты
validation получены после выбора конфигурации.

### Подбор порогов

| Поиск | Диапазон / шаг | Троек | Лучшие train-пороги | Train BS score | Held-out BS score |
| --- | --- | ---: | --- | ---: | ---: |
| coarse | -0.10..0.70 / 0.02 | 10 660 | 0.100 / 0.200 / 0.400 | 0.2357 | 0.2414 |
| fine | 0.020..0.500 / 0.005 | 147 440 | **0.100 / 0.205 / 0.390** | **0.2363** | **0.2424** |

Точный перебор реализован через накопленные распределения dNBR для каждого
истинного класса. Это эквивалентно оценке каждой тройки на всех train-пикселях,
но не требует повторно читать миллионы пикселей для каждой конфигурации.

### Итог лучшей train-конфигурации

Пороги: `t0=0.100`, `t1=0.205`, `t2=0.390`.

| Набор | IoU_burn | IoU class 1 | IoU class 2 | IoU class 3 | mIoU_sev | BS weighted score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.3628 | — | — | — | 0.3646 | 0.2363 |
| held-out event validation | **0.3737** | 0.1805 | 0.3601 | 0.5760 | **0.3722** | **0.2424** |

### Вывод

dNBR действительно содержит сигнал выгорания: результат стабильно близок на
train и на новых fire events, а сильное поражение выделяется лучше остальных
классов. Однако `IoU_burn≈0.374` и особенно `IoU class 1≈0.180` недостаточны
для финального решения BS. В отличие от AF thermal baseline, здесь rule-based
dNBR следует использовать как воспроизводимый baseline и дополнительный
признак/приор для ML, а не как окончательную систему.

Скрипт эксперимента: `scripts/evaluate_bs_dnbr_baseline.py`. Полный JSON
каждого запуска сохраняется в `ml/artifacts/` и не коммитится:

```powershell
python scripts\build_ml_split.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --kind bs --output ml\artifacts\bs_event_split_seed42.json --seed 42

python scripts\evaluate_bs_dnbr_baseline.py `
  --data-dir D:\ForestFire\fire-train-renamed `
  --split ml\artifacts\bs_event_split_seed42.json `
  --min-threshold 0.02 --max-threshold 0.50 --step 0.005 `
  --top-k 15 --output ml\artifacts\bs_dnbr_seed42_grid005.json
```

