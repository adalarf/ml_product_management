Проект с домашними заданиями по курсу "Управление ML-продуктом на всех этапах жизненного цикла"

# Установка и запуск

1. Склонировать проект `git clone https://github.com/adalarf/ml_product_management.git`
2. Установка окружения `make setup`
3. Проверка кода `make check`

Нужны `uv`, `make` и Docker (с Docker Compose).

# Данные

Датасет CIFAR-10. Сырой архив лежит в S3 без изменений, обработанный датасет собирается из него одной командой и выкладывается обратно в S3 под версией.

## Хранилище

```
make s3-up      # поднять хранилище (RustFS) и создать бакет
make s3-down    # остановить (данные остаются в volume)
```

Перед первым `make s3-up` создайте `.env` из примера: `cp .env.example .env` (при необходимости поменяйте значения).
Логин, пароль, адрес и имя бакета берутся из `.env`

Посмотреть содержимое:

- веб-консоль RustFS: <http://localhost:9001> (логин и пароль: `S3_ACCESS_KEY` и `S3_SECRET_KEY` из `.env`);
- AWS CLI: `AWS_ACCESS_KEY_ID=<S3_ACCESS_KEY> AWS_SECRET_ACCESS_KEY=<S3_SECRET_KEY> aws --endpoint-url http://localhost:9000 s3 ls s3://ml-product-management/ --recursive`.

Данные хранятся в именованном volume `rustfs_data` и переживают `make s3-down` и повторный
`make s3-up`

## Как запускать

```
make setup      # окружение
cp .env.example .env   # настройки S3 (один раз)
make s3-up      # хранилище и бакет
make ingest     # скачать архив CIFAR-10 с сайта и положить в S3 (raw/)
make data       # весь конвейер обработки, выгрузка версии v1 в S3 (processed/)
```

## Шаги конвейера

| Make-цель | Команда | Что делает |
|---|---|---|
| `make ingest` | `mlpm-ingest` | скачивает архив с сайта через torchvision и кладёт в S3 `raw/` (существующий объект не перезаписывается) |
| `make download-from-s3` | `mlpm-download-from-s3` | скачивает сырой архив из S3 в `data/raw/` |
| `make convert-to-safe` | `mlpm-convert-to-safe` | pickle -> PNG + JSON-разметка в `data/interim/cifar10/` |
| `make partitioning` | `mlpm-partitioning` | стратифицированное разбиение train/val (раздел `partitioning`) |
| `make stats` | `mlpm-stats` | среднее и std по каналам (только train) в `stats.json` |
| `make export-to-s3` | `mlpm-export-to-s3` | выгрузка версии в S3 `<prefix>/<version>/` из раздела `export` в `configs/s3.yaml`, по умолчанию `processed/cifar10/v1/` |

`make data` выполняет шаги с `download-from-s3` по `export-to-s3`. Переменная Makefile
`ENV_FILE` задаёт файл с настройками S3 (по умолчанию `.env`).


## Что лежит в S3

Бакет `ml-product-management` (имя из `S3_BUCKET`):

```
raw/
    cifar-10-python.tar.gz          # исходный архив, как скачан с сайта
processed/cifar10/v1/
    images/                         # PNG: train_00000.png ... test_09999.png
    annotations/
        train.json                  # 45 000 изображений
        val.json                    # 5 000 изображений
        test.json                   # 10 000 изображений
    stats.json                      # mean и std по каналам (только train)
    manifest.json                   # число изображений, размеры и sha256 файлов,
                                    # параметры (размер валидации, seed), git-коммит
```

## Карточка датасета

- **Источник:** CIFAR-10, <https://www.cs.toronto.edu/~kriz/cifar.html>,
  Python-версия `cifar-10-python.tar.gz`
- **Размер:** 60 000 цветных изображений 32×32, 10 классов, 6 000 на класс; 50 000 в исходном train и 10 000 в test.
- **Классы:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- **Формат после обработки:** PNG, RGB, 8 бит на канал; разметка в JSON (`image_id`, `file_name`,
  `label`, `label_name`, `sha256`).
- **Разбиение:** из исходного train выделена валидация 5 000 (по 500 на класс), в train остаётся 45 000 (по 4 500 на класс); стратификация по классам, seed 42. Исходный test не меняется.
Изображение при разбиении сохраняет свой `image_id` (`train_...`), меняется только файл разметки.
- **Известные проблемы:**
  - низкое разрешение (32×32);
  - в тестовой выборке есть почти-дубликаты обучающих изображений (по оценке авторов ciFAIR, порядка 3%), поэтому метрики на test могут быть слегка завышены;
  - данные не в git: каталог `data/` в `.gitignore`, источник правды — S3.
