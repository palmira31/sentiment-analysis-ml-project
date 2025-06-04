# Sentiment Analysis Project

## Project Overview

Задача – разработать модель анализа эмоциональной окраски текста, способную
классифицировать предложения на три категории: позитивные, нейтральные и
негативные. Задача является актуальной, так как подобный анализ текста широко
используется в социальных сетях, службах поддержки клиентов и маркетинговых
исследованиях, например, для оценки общественного мнения о брендах и продуктах,
отслеживания реакции аудитории на актуальные события и новости и тд

### Модели

В проекте реализованы две модели:

1. **Базовая модель** ([linear_model.py](project/my_modules/linear_model.py)):

   - Полносвязная нейронная сеть
   - TF-IDF векторизация + линейные слои

2. **Основная модель** ([rnn_model.py](project/my_modules/rnn_model.py)):
   - Рекуррентная нейронная сеть (RNN)
   - Лучше учитывает последовательную природу текста

### Метрики

- Loss (функция потерь)
- Accuracy (точность классификации)
- F1-score (для оценки качества на несбалансированных данных)
- Confusion Matrix (матрица ошибок)

Все метрики доступны в MLflow UI во время обучения.

### Артефакты

Для каждой обученной модели сохраняются следующие артефакты:

- Веса модели в формате PyTorch (`.pt`)
- Экспортированная ONNX-модель для быстрого инференса
- Предобработка данных:
  - TF-IDF векторизатор
  - Кодировщик меток (LabelEncoder)

Все артефакты автоматически версионируются через MLflow и доступны для
воспроизведения результатов или запуска инференса. Также добавлен скрипт для
конвертации моделей в TensorRT формат для ускорения инференса.

### Версионирование данных и моделей

Проект использует DVC (Data Version Control) для управления:

- Датасетом (папка `Data/`)
- Обученными моделями
- Артефактами предобработки

## Project Structure

```
├── Data/                   # Data directory (gitignored)
│   └── 1/                 # Default Kaggle dataset directory
├── conf/                   # Configuration files
│   ├── config.yaml        # Main configuration file
│   ├── convert/           # Model conversion configs
│   ├── data_loading/      # Data loading configs
│   ├── inference/         # Inference configs
│   ├── logging/          # Logging configs
│   ├── model/            # Model-specific configs
│   └── training/         # Training configs
├── model_checkpoints/     # Saved model checkpoints
├── plots/                # Project artifacts
│   ├── mlruns/          # MLflow artifacts and metrics
│   └── outputs/         # Hydra outputs and configs
├── project/              # Main project code
│   └── my_modules/      # Core implementation modules
│       ├── data.py      # Data loading and processing
│       ├── train.py     # Training script
│       ├── infer.py     # Inference script
│       ├── linear_model.py  # Linear model implementation
│       └── rnn_model.py    # RNN model implementation
├── scripts/             # Utility and execution scripts
│   ├── data/           # Data processing scripts
│   ├── download_data.py # Data download script
│   └── convert_to_tensorrt.py  # Model conversion script
├── test_results/       # Inference results
└── tests/             # Test files
```

## Requirements

- Python >= 3.9
- Dependencies are managed using Poetry

### Main Dependencies

- PyTorch (>=2.7.0)
- PyTorch Lightning (>=2.5.1)
- pandas (>=2.2.3)
- scikit-learn (>=1.6.1)
- tensorboard (>=2.19.0)
- kagglehub (>=0.3.10)

## Setup

1. Клонируйте репозиторий

```bash
git clone https://github.com/palmira31/sentiment-analysis-ml-project.git
cd sentiment-analysis-ml-project
```

2. Установите poetry (если не установлен)
3. Установите зависимости проекта

```bash
poetry install
```

## Train

Для работы с проектом нужно использовать два терминала:

- Первый терминал будет постоянно занят процессом MLflow UI
- Второй терминал будет использоваться для запуска обучения и других команд

### 1. Загрузка данных

Данные для обучения и теста загружаются из датасета kaggle. Здесь и далее
запускать скрипты можно либо непосредственно обращаясь к ним, либо с помощью
скрипта commands.py, где реализованы функции запуска основных скриптов проекта.
Запуск скрипта:

```bash
poetry run python scripts/download_data.py
```

Запуск с помощью commands.py:

```
poetry run commands.py download-data
```

### 2. Запуск MLflow UI (Терминал 1)

Откройте первый терминал и запустите MLflow UI:

```bash
poetry run python commands.py ui
```

Это:

- Создаст директории `plots` и `plots/mlruns` если они не существуют
- Запустит MLflow UI на порту 8080
- **Важно**: оставьте этот терминал открытым! Не закрывайте его и не используйте
  для других команд

После запуска откройте в браузере: http://127.0.0.1:8080

Для запуска без помощи скрипта введите следующую команду в отдельном терминале:

```
mlflow ui --backend-store-uri plots/mlruns --port 8080
```

Внимание, в этом случае при отсутсвии папко plots и внутри нее mlruns, их
необходимо создать вручную!

### 2. Обучение модели (Терминал 2)

Откройте второй терминал и запустите обучение:

```bash
# Обучение с параметрами по умолчанию
poetry run python commands.py train

# Или с указанием параметров
poetry run python commands.py train --model rnn --experiment-name rnn_exp_01 --max-epochs 5
```

Параметры обучения:

- `--model`: тип модели (linear/rnn)
- `--experiment-name`: название эксперимента для MLflow
- `--max-epochs`: количество эпох обучения

Во время обучения вы можете наблюдать за метриками в веб-интерфейсе MLflow:

- loss на валидации и обучении
- accuracy и другие метрики
- используемые гиперпараметры
- git commit id текущей версии кода

После обучения все артефакты сохраняются в директорию
`model_checkpoints/<experiment_name>/`: | Файл | Назначение |
|------------------|------------| | `model.pt` | Веса обученной модели в формате
PyTorch | | `model.onnx` | Модель, экспортированная в ONNX-формат | |
`vectorizer.pkl` | TF-IDF vectorizer для преобразования текста | |
`label_encoder.pkl` | LabelEncoder для преобразования меток | | `epoch=...ckpt`
| Чекпоинт модели от PyTorch Lightning |

### 3. Мониторинг

В любой момент вы можете проверить статус всех компонентов проекта:

```bash
poetry run python commands.py status
```

## Инференс

Для использования обученной модели на новых данных:

```bash
poetry run python commands.py infer
```

```bash
poetry run python commands.py infer --checkpoint-path "model_checkpoints/rnn_model_03/model.onnx" --output-dir "test_results"
```

Скрипт автоматически:

1. Загрузит все необходимые файлы (vectorizer.pkl, label_encoder.pkl,
   model.onnx)
2. Применит предобработку к новым данным
3. Запустит инференс через ONNX
4. Сохранит результаты в указанную директорию

если сделаю inference server, то добавить инфу и по нему!!!!!!!!!!!!

## Настройка конфигов

Все настройки конфигурации хранятся в папке conf/. Используется библиотека Hydra
для управления конфигурацией. Примеры конфигов:

- conf/config.yaml – основной конфиг
- conf/model/linear.yaml – параметры линейной модели
- conf/logging/logging.yaml – настройки MLflow и путей
- conf/training/training.yaml – параметры обучения

## Дополнительная информация

- С Kaggle автоматически файлы сохраняются в папку "1", поэтому в конфигах путь
  указан как ../Data/1
- Для работы с проектом рекомендуется использовать Python 3.9-3.12 (Python 3.13
  не поддерживается из-за несовместимости зависимостей, а предпочительнее всего
  3.10)
- Если при запуске pre-commit возникает ошибка SSL, самый рабочий вариант это
  обновить conda
- В папку plots сложила логи и данные для графиков, именно так я
  интерпретировала заданее

---

## License

See the LICENSE file for details.

## Author

Irina Pavlova

---
