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

1. Клонируйте репозиторий:

```bash
git clone https://github.com/palmira31/sentiment-analysis-ml-project.git
cd sentiment-analysis-ml-project
```

2. Создайте и активируйте conda окружение:

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```

3. Установите poetry (если не установлен):
4. Установите зависимости проекта:

```bash
poetry install
```

**Важно**: Рекомендуется использовать Python 3.10, так как он обеспечивает
наилучшую совместимость со всеми зависимостями проекта. Python 3.13 не
поддерживается из-за несовместимости некоторых пакетов.

## Train

Для работы с проектом нужно использовать два терминала:

- Первый терминал будет постоянно занят процессом MLflow server
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

### 2. Запуск MLflow сервера (Терминал 1)

1. Откройте первый терминал и запустите MLflow сервер:

```bash
python -m mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlruns --default-artifact-root ./artifacts
```

После запуска:

- MLflow UI будет доступен по адресу: http://127.0.0.1:8080
- Метаданные экспериментов сохраняются в `./mlruns/`
- Артефакты моделей сохраняются в `./artifacts/`

При запуске обучения создаются:

1. В директории `plots/YYYY-MM-DD/HH-MM-SS/`:

   - `loss_curves.png` - графики функций потерь
   - `accuracy.png` - график точности на валидации
   - `confusion_matrix.png` - матрица ошибок
   - `.hydra/` - конфигурация запуска

2. В директории `artifacts/experiment_name/run_id/`:

   - `model.pt` - веса модели PyTorch
   - `vectorizer.pkl` - обученный векторайзер
   - `label_encoder.pkl` - энкодер меток

3. В директории `mlruns/`:
   - Метаданные экспериментов
   - Метрики обучения
   - История запусков

В веб-интерфейсе MLflow (http://127.0.0.1:8080) доступно:

- Графики метрик в реальном времени
- История всех экспериментов
- Сравнение разных запусков
- Параметры обучения
- Git commit ID для воспроизводимости

После обучения модель можно:

1. Загрузить через MLflow API
2. Использовать для инференса
3. Развернуть через MLflow serving

Для запуска MLflow сервера также можно использовать команду:

```bash
poetry run python commands.py ui
```

Важные замечания:

- Оставьте терминал с запущенным сервером открытым
- Для остановки сервера используйте Ctrl+C

После запуска MLflow и обучения модели создаются следующие директории:

```
MLOps/
├── artifacts/            # MLflow артефакты моделей
│   └── experiment_name/  # Организовано по экспериментам
│       └── run_id/      # Артефакты конкретного запуска
├── mlruns/              # Метаданные и метрики MLflow
└── plots/               # Графики и логи обучения (как требуется в задании)
    └── YYYY-MM-DD/      # Организовано по дате
        └── HH-MM-SS/    # Результаты конкретного запуска
            ├── loss_curves.png
            ├── accuracy.png
            ├── confusion_matrix.png
            └── .hydra/  # Конфигурация запуска
```

### 3. Обучение модели (Терминал 2)

Откройте второй терминал и запустите обучение:

```bash
# Обучение с параметрами по умолчанию
poetry run python commands.py train

# Или с указанием параметров
poetry run python commands.py train --model rnn --experiment-name rnn_exp_01 --max-epochs 5

# Запуск напрямую через poetry run python
poetry run python project/my_modules/train.py

# Или с указанием параметров
poetry run python project/my_modules/train.py model=rnn experiment_name=rnn_exp_01 max_epochs=5
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

### 4. Мониторинг

В любой момент вы можете проверить статус всех компонентов проекта:

```bash
poetry run python commands.py status
```

## Инференс

### Локальный инференс

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

### Инференс через MLflow

После обучения модель автоматически регистрируется в MLflow Model Registry. Для
использования модели через MLflow:

1. Запустите MLflow сервер (если еще не запущен):

```bash
poetry run python commands.py ui
```

2. Запустите serving модели (в новом терминале):

```bash
mlflow models serve -m "models:/sentiment_classifier_rnn/latest" -p 5001
```

3. Теперь можно делать запросы к модели через API:

```bash
curl -X POST -H "Content-Type:application/json" --data '{"inputs": ["This is a great movie!", "I did not like this film"]}' http://127.0.0.1:5001/invocations
```

Преимущества использования MLflow для инференса:

- Автоматическое версионирование моделей
- REST API для простой интеграции
- Масштабируемость и возможность развертывания в production
- Отслеживание всех версий моделей и их параметров

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
