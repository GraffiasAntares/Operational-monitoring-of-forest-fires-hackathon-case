# Мониторинг природных пожаров

Репозиторий содержит два независимых модуля соревнования: Active Fire (AF) и
Burn Severity (BS), а также веб-приложение «Лесной дозор» на FastAPI,
React и Leaflet. AF определяет активные пожары по VIIRS, BS классифицирует
повреждение территории по паре спутниковых наблюдений до/после.

## Окружение

Нужны Python 3.11+ (локально проверялся 3.12), PyTorch 2.5+,
для фронтенда — Node.js 22.12+ и npm. BS и общий инференс поддерживают CPU;
скрипт обучения AF Direct требует NVIDIA GPU и CUDA-сборку PyTorch.
Команды выполняются из корня репозитория в PowerShell, если не указано другое.
Создайте окружение и установите зависимости из `pyproject.toml`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -c "import subprocess,sys,tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; subprocess.check_call([sys.executable,'-m','pip','install',*p['dependencies'],*p['optional-dependencies']['ml'],*p['optional-dependencies']['dev']])"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

Данные должны находиться в `DATA_DIR/train/bs/` в выданной организаторами
структуре. `DATA_DIR` и `TEST_DATA_DIR` в примерах замените своими путями.
Для обучения AF проверка CUDA должна вернуть `True`; иначе установите
CUDA-сборку PyTorch, совместимую с драйвером. Локальный инференс по GeoTIFF
не требует ключей внешних API.

## BS baseline и обучение

```powershell
New-Item -ItemType Directory -Force ml/artifacts
python scripts/build_ml_split.py --data-dir DATA_DIR --kind bs --output ml/artifacts/bs_split_seed42.json
python scripts/fit_bs_preprocessing.py --data-dir DATA_DIR --split ml/artifacts/bs_split_seed42.json --output ml/artifacts/bs_preprocessing_seed42.npz
python scripts/run_bs_baseline.py --data-dir DATA_DIR --split ml/artifacts/bs_split_seed42.json --output-dir ml/artifacts/dnbr_baseline_seed42
python train_bs.py --data-dir DATA_DIR --split ml/artifacts/bs_split_seed42.json --preprocessing ml/artifacts/bs_preprocessing_seed42.npz --output-dir ml/artifacts/bs_unet_full_seed42 --feature-mode full
```

Скрипт сохраняет лучший чекпойнт в `best.pt`, историю обучения и фиксированную
конфигурацию. Он также сохраняет `checkpoints/step_*.pt` каждые 500 batch-шагов,
`checkpoints/epoch_*.pt` каждые 5 эпох и итоговую модель `last.pt`. Интервалы
меняются параметрами `--checkpoint-every-steps` и `--checkpoint-every`.
Используются group split по `fire_event_id`, mixed precision на CUDA,
`batch_size=1`, накопление градиента до batch 4 и ранняя остановка по локальному
BS score.

## Инференс BS

```powershell
python predict_bs.py --data-dir TEST_DATA_DIR --checkpoint ml/artifacts/bs_unet_full_seed42/best.pt --preprocessing ml/artifacts/bs_preprocessing_seed42.npz --output predictions_bs.npz
```

`TEST_DATA_DIR` может содержать `test/bs`, `bs` или сразу каталог BS с
`meta.csv`. Выходной NPZ содержит одну маску `0..3` на чип. Для BS-only
шаблона можно дополнительно передать `--template` и `--submission`; объединённый
конкурсный CSV формируется после добавления AF-предсказаний и проверяется так:

```powershell
python scripts/validate_submission.py --template sample_submission.csv --submission submission.csv
```

## Проверка чекпойнта на validation

Например, после 15-й эпохи можно проверить сохранённый чекпойнт, не
останавливая обучение:

```powershell
python evaluate_bs.py --data-dir . --split ml/artifacts/bs_split_seed42.json --preprocessing ml/artifacts/bs_preprocessing_seed42.npz --checkpoint ml/artifacts/bs_unet_full_seed42/checkpoints/epoch_015.pt --output-dir ml/artifacts/eval_epoch_015 --device cuda
```

В примере оценки замените `--data-dir .` на корень своего обучающего набора;
для CPU укажите `--device cpu`.

## Схема пайплайна и исходный код

```text
Обучение BS: train/bs → групповой split → статистики train → U-Net → best.pt
Обучение AF: train/af → strict split и нормализация → ResUNet → best.pt

inference.py:
test/af → VIIRS → thermal I4 > 330 K (по умолчанию)
        → VIIRS + AUX → AF ResUNet (при --af-checkpoint)
test/bs → Sentinel-2/1 до/после + AUX → 74 признака → BS U-Net → классы 0–3
Маски AF/BS + test/sample_submission.csv → RLE → submission.csv

Веб: React/Leaflet → FastAPI → NASA FIRMS → точки / 10 интервалов
                            → Copernicus → снимки начала/конца → BS-маска
```

| Путь | Назначение |
| --- | --- |
| `inference.py` | Единая точка входа AF + BS и запись конкурсного CSV |
| `train_bs.py`, `predict_bs.py`, `evaluate_bs.py` | Обучение, инференс, оценка BS |
| `ml/bs/` | Данные, предобработка, модель, loss, метрики, калибровка |
| `ml/af/`, `ml/af_direct.py`, `ml/af_residual.py` | Данные AF, ResUNet и признаки |
| `scripts/` | Подготовка данных, split, обучение AF и проверка submission |
| `ml/submission.py` | Формат CSV и кодирование RLE |
| `backend/app/main.py` | FastAPI; маршруты в `api/`, адаптеры и модели в `services/` |
| `backend/app/core/config.py`, `.env.example` | Настройки API и путей весов |
| `frontend/src/App.tsx`, `frontend/src/api.ts` | Интерфейс и запросы к backend |
| `pyproject.toml`, `frontend/package-lock.json` | Зависимости |
| `compose.yaml`, `deploy/nginx.conf` | Контейнерный запуск и прокси |

## Структура входных данных

Используйте архивы организаторов без переименования файлов. Демонстрационные
загрузчики спутниковых данных не заменяют конкурсный набор.

```text
DATA_DIR/train/
  af/
    meta.csv
    viirs/<chip_id>_VIIRS_I1-I5.tif
    aux/<chip_id>_AUX.tif
    masks/<chip_id>_MASK.tif
  bs/
    meta.csv
    sentinel2_pre/<chip_id>_Sentinel-2_pre.tif
    sentinel2_post/<chip_id>_Sentinel-2_post.tif
    sentinel1_pre/<chip_id>_Sentinel-1_pre.tif
    sentinel1_post/<chip_id>_Sentinel-1_post.tif
    aux/<chip_id>_AUX.tif
    masks/<chip_id>_MASK.tif

TEST_DATA_DIR/test/
  meta.csv
  sample_submission.csv
  af/                         # viirs/ и aux/ с теми же суффиксами файлов
  bs/                         # все пять входных каталогов BS, без masks/
```

В метаданных обязателен уникальный `chip_id`; общий `test/meta.csv` также
содержит `kind` (`af` или `bs`). BS split требует `fire_event_id`.
AF strict split использует исходные `acq_datetime`, `epsg`, `x_min`, `x_max`,
`y_min`, `y_max`: сохраняйте всю исходную метаинформацию.

Растры одного чипа должны совпадать по сетке, размерам, CRS и transform.
AF: 8 каналов VIIRS + 5 AUX, затем 26 признаков Direct-модели.
BS: по 10 каналов Sentinel-2 (9 спектральных + SCL), по 2 Sentinel-1,
3 AUX — всего 27 исходных и 74 после предобработки `full`.
Порядок каналов должен соответствовать организаторскому формату, RGB PNG
вместо этих TIFF не подходит.

Маски: AF `0/1`, BS `0/1/2/3` (фон / низкая / средняя / высокая тяжесть),
значение `255` исключается из loss и метрик. Стандартные конкурсные размеры:
AF 256×256, BS 512×512. Тестовые эталонные маски не требуются.

## Обучение AF Direct

Thermal baseline не требует обучения. Для нейросетевого AF на CUDA:

```powershell
python scripts/prepare_af_direct_data.py --data-dir DATA_DIR --output-dir ml/artifacts/af_direct_strict_v1
python scripts/train_af_direct.py --data-dir DATA_DIR --prepared-dir ml/artifacts/af_direct_strict_v1 --output-dir ml/artifacts/af_direct_strict_v1/checkpoints --batch-size 4 --accumulation 2 --epochs 70 --patience 12 --seed 42
python scripts/evaluate_af_direct.py --data-dir DATA_DIR --prepared-dir ml/artifacts/af_direct_strict_v1 --checkpoint ml/artifacts/af_direct_strict_v1/checkpoints/best.pt
```

Подготовка создаёт `manifest.csv` и `normalization.json`, не копируя TIFF.
Checkpoint содержит веса, нормализацию и порог. При нехватке GPU-памяти
используйте `--batch-size 2 --accumulation 4`.

## Получение и размещение весов

`ml/artifacts/` исключён из Git: после клонирования веса нужно получить
от автора поставки либо обучить командами выше. Автоматического скачивания
весов и публичной ссылки на них в проекте нет. Используйте доверенные `.pt`.

Пути по умолчанию для общего CLI-инференса и веб-сервиса:

```text
ml/artifacts/
  bs_preprocessing_seed42.npz
  bs_unet_full_tuned_seed42/
    best.pt
    config.json                 # сохранённые параметры обучения
    logit_calibration.json      # необязательно, для общего CLI
    dnbr_hybrid.json             # необязательно, для общего CLI
  af_direct_strict_v1/checkpoints/best.pt   # необязательно
```

Пример обучения BS выше сохраняет новый эксперимент в `bs_unet_full_seed42`,
а не перезаписывает готовый `bs_unet_full_tuned_seed42`. Для своих весов
передайте путь явно, как в команде ниже, либо измените `--output-dir` обучения.
BS checkpoint и файл статистик `.npz` должны быть из одного комплекта.
Архитектура также хранится внутри checkpoint. Веб-сервис принимает другие
пути через `BS_MODEL_PATH` и `BS_PREPROCESSING_PATH` в `.env`.

## Общий инференс AF + BS и submission.csv

После обучения BS по инструкции выше:

```powershell
python inference.py --data-dir TEST_DATA_DIR --output outputs/submission.csv --bs-checkpoint ml/artifacts/bs_unet_full_seed42/best.pt --bs-preprocessing ml/artifacts/bs_preprocessing_seed42.npz --bs-logit-calibration ml/artifacts/bs_unet_full_seed42/logit_calibration.json --bs-dnbr-hybrid ml/artifacts/bs_unet_full_seed42/dnbr_hybrid.json --device auto --masks-output outputs/predictions.npz
python scripts/validate_submission.py --template TEST_DATA_DIR/test/sample_submission.csv --submission outputs/submission.csv
```

**Результат — `outputs/submission.csv` относительно корня репозитория.**
Каталог создаётся автоматически; путь задаётся `--output`.
`outputs/predictions.npz` — необязательный архив масок по `chip_id`.
Автоматической отправки на конкурсную платформу нет.

Для готового комплекта в стандартных путях и нейросетевого AF:

```powershell
python inference.py --data-dir TEST_DATA_DIR --output outputs/submission_af_bs.csv --af-checkpoint ml/artifacts/af_direct_strict_v1/checkpoints/best.pt --device auto
```

Без `--af-checkpoint` используется thermal baseline `I4 > 330 K` с valid-маской.
На CPU укажите `--device cpu`. CSV содержит `chip_id,class_id,rle`, порядок
строк соответствует шаблону. Класс 0 не записывается; RLE построчный, от 1.

Общий CLI применяет калибровку logits и включённый dNBR hybrid, если JSON
существуют по указанным путям. Без этих необязательных файлов используется
обычный BS U-Net `argmax`. JSON не создаются самим `train_bs.py`; для них есть
`scripts/calibrate_bs_logits.py` и `scripts/evaluate_bs_unet_dnbr_hybrid.py`.
Не используйте калибровку чужого checkpoint для нового эксперимента.

## Запуск веб-приложения без Docker

Установите Python-зависимости и разместите BS-веса. Если `.env` отсутствует:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Заполните `FIRMS_MAP_KEY`, `COPERNICUS_CLIENT_ID`, `COPERNICUS_CLIENT_SECRET`;
адреса получения ключей указаны в `.env.example`. Не публикуйте ключи.
Для весов из примера обучения добавьте в `.env`:

```dotenv
BS_MODEL_PATH=ml/artifacts/bs_unet_full_seed42/best.pt
BS_PREPROCESSING_PATH=ml/artifacts/bs_preprocessing_seed42.npz
```

В первом терминале из корня проекта с активированным окружением:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

API: `http://127.0.0.1:8000/docs`, health: `http://127.0.0.1:8000/health`.
Health проверяет запуск сервера, но не наличие весов и доступность внешних API.

Во втором терминале:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Интерфейс: `http://127.0.0.1:5173`. Vite проксирует `/api` на порт 8000.
Остановка — `Ctrl+C` в каждом терминале. После смены весов перезапустите
backend: модель кэшируется в памяти.

### Текущее поведение и ограничения

- Загрузка запускается только кнопкой. Оперативный режим получает точки
  NASA FIRMS за один день для области, а не запускает конкурсную AF-модель.
- Архивный период минимум из 10 календарных дней делится на 10 непрерывных
  интервалов с разницей длины не больше дня.
- Снимки отображаются только на первом и последнем шагах. Источник снимков —
  **Copernicus Sentinel-2**, источник точек — **NASA FIRMS**.
- Выбираются разные ближайшие даты в окне ±7 дней с облачностью ≤30% по
  метаданным сцены. Это не гарантирует безоблачность и полное покрытие области.
  Фактическая дата возвращается в заголовке API `X-Actual-Scene-Date`.
- Слой «Изменения dNBR (BS-модель)» использует BS U-Net; маска отображается
  поверх конечного снимка. Все три цвета возможны, но не обязательны.
- Веб-BS получает реальные Sentinel-2, но SAR и AUX пока заполняются нулями.
  Для больших областей разрешение снижается. Это демонстрационный режим,
  не эквивалент полного конкурсного инференса.
- Границы задаются как `west,south,east,north`. Отсутствие подходящих сцен,
  ключей, весов или доступности API приводит к ошибке; в UI есть повтор запроса.

## Docker Compose (необязательно)

Нужны Docker Desktop с Linux-контейнерами, заполненный `.env` и веса
в стандартных путях `bs_unet_full_tuned_seed42/best.pt` и
`bs_preprocessing_seed42.npz` внутри `ml/artifacts/`:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f backend
```

Интерфейс: `http://localhost:8080`; порт меняется через `APP_PORT` в `.env`.
Backend работает на CPU, frontend обслуживает Nginx. Веса монтируются только
для чтения; ключи не включаются в образы. Compose явно задаёт пути весов
внутри `/models`; для иного комплекта измените эти пути в `compose.yaml`.
Остановка: `docker compose down`. Подробнее: [docs/DOCKER.md](docs/DOCKER.md).
Конфигурация подготовлена; сборка и запуск Docker при последнем обновлении
не выполнялись.

## Проверки и документация

```powershell
python -m pytest -q
cd frontend
npm run build
```

API-тесты используют заглушки и не заменяют проверку реальных запросов.
Параметры скриптов доступны через `--help`.

- [Формат данных, метрики и submission](docs/ML_DATA_AND_SUBMISSION.md).
- [Описание конкурсного пайплайна](docs/CURRENT_SOLUTION_PIPELINE.md).
- [Руководство AF Direct](docs/AF_DIRECT_RESUNET_GUIDE.md).
- [Стратегия BS](docs/BS_MODEL_STRATEGY.md).

Документы экспериментов содержат исторические пути и результаты;
для текущих команд запуска используйте этот README и `--help` скриптов.
