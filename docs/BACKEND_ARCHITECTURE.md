# Backend: структура и API

## Назначение

Backend — FastAPI-сервис MVP для мониторинга лесных пожаров. Он возвращает
GeoJSON, поэтому клиент может напрямую рисовать все пространственные сущности
на карте. Сервис не хранит события в базе: на текущем этапе он получает и
вычисляет данные по параметрам запроса.

## Структура проекта

```text
backend/
  app/
    main.py                         # FastAPI-приложение и регистрация роутов
    core/config.py                  # .env и настройки внешних сервисов
    api/
      hotspots.py                   # FIRMS: активные очаги
      events.py                     # HDBSCAN: пожарные события
      burned_areas.py               # Copernicus: подтверждённая гарь
      dnbr.py                       # Sentinel-2: тяжесть dNBR
      forest_impact.py              # WorldCover: ущерб лесу
      safe.py                       # статус доступности SAFE
    models/                         # Pydantic-модели API-ответов
    services/                       # клиенты внешних API и геообработка
  tests/                            # unit/API-тесты
docs/                               # проектная документация
```

## Конфигурация

Локальный файл `.env` автоматически читается при запуске. Он не должен
попадать в Git.

```ini
FIRMS_MAP_KEY=...
COPERNICUS_CLIENT_ID=...
COPERNICUS_CLIENT_SECRET=...
# Необязательно: путь к совместимым весам SAFE в будущем
SAFE_MODEL_PATH=...
```

## Общие правила API

- Базовый префикс: `/api/v1`.
- География: WGS84, координаты `[longitude, latitude]`.
- `bbox` всегда передаётся как `west,south,east,north`.
- Пространственные ответы имеют формат `FeatureCollection`.
- Ошибка `422` означает некорректные параметры; `502` — недоступность внешнего
  поставщика; `503` — не настроены ключи.

## Endpoints

### Состояние сервиса

`GET /health`

Ответ: `{"status": "ok"}`.

### Активные очаги

`GET /api/v1/hotspots?bbox=92.5,55.8,93.2,56.4&days=1`

Источник — NASA FIRMS VIIRS SNPP и NOAA-20. Возвращает точечный GeoJSON;
в `properties` доступны источник, время, confidence, FRP и brightness.
Исторический режим: добавить `start_date=YYYY-MM-DD`.

### Пожарные события

`GET /api/v1/fire-events?bbox=92.5,55.8,93.2,56.4&days=1&min_cluster_size=3`

Сначала запрашивает FIRMS, затем HDBSCAN объединяет точки в пространстве и
времени. Каждая feature — точка-центр события. В `properties` находятся:
`bbox`, `hotspot_count`, `first_seen`, `last_seen`, `max_frp`,
`mean_confidence`, `source_counts`.

### Подтверждённая гарь

`GET /api/v1/burned-areas?bbox=-118.8,33.6,-118.3,34.3&date=2025-01-10`

Источник — Copernicus Global Burnt Area V4, 300 м. Возвращаются полигоны
подтверждённой гари. В `properties`: `area_ha`, `day_of_burn`,
`burned_fraction`, `confidence`, `pixel_count`, `resolution_m`.

### Тяжесть повреждения dNBR

`GET /api/v1/severity/dnbr?bbox=...&before_date=2025-01-01&after_date=2025-01-20&window_days=7`

Берёт B08/B12 Sentinel-2 L2A в окнах до и после пожара, исключает облачные
пиксели по SCL и возвращает полигоны классов `low`, `moderate`, `high`.
В `properties`: `severity`, `mean_dnbr`, `area_ha`, `before_date`,
`after_date`, `resolution_m`.

### Ущерб лесу

`GET /api/v1/forest-impact?bbox=...&date=2025-01-10`

Получает контуры гари, затем пересекает их с ESA WorldCover 2021 v200.
В каждой feature добавляется `forest_burned_ha`; в metadata — суммарное
`forest_burned_ha`. Из-за 10-метрового слоя bbox должен быть небольшим.

### SAFE

`GET /api/v1/safe/status`

Показывает, подключены ли совместимые веса SAFE. Пока без весов это статусный
endpoint, а не сегментация.

## Запуск и тестирование

```powershell
python -m uvicorn backend.app.main:app --reload
python -m pytest -q
```

Интерактивная спецификация FastAPI доступна на `http://127.0.0.1:8000/docs`.
