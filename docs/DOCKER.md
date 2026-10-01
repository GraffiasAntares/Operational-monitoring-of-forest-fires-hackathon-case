# Запуск Docker

Состояние приложения восстановлено до внедрения проекта коллеги: `cf96795`.
Docker запускает backend (Python 3.12, CPU PyTorch) и frontend (Nginx).

Необходимы Docker Desktop с Linux-контейнерами, `.env` с ключами FIRMS/Copernicus
и локальные файлы модели:

- `ml/artifacts/bs_unet_full_tuned_seed42/best.pt`
- `ml/artifacts/bs_preprocessing_seed42.npz`

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f backend
```

Открыть http://localhost:8080. Другой порт: задать `APP_PORT` в `.env`.
API проксируется через Nginx, порт бэкенда наружу не публикуется.
Ключи передаются при запуске; `.env`, датасеты и веса не включаются в образы.
Веса подключаются из локального `ml/artifacts` только для чтения.

Остановка: `docker compose down`.
Проверка: `docker compose exec backend python -c "import torch; print(torch.__version__)"`.

После отката отдельно исправлен запрос каталога спутниковых сцен:
используется поддерживаемый Copernicus фильтр CQL2 вместо старого `query`.
Docker сам логику приложения не меняет.
Копия отменённых изменений сохранена в `tmp/before-colleague-rollback`.
