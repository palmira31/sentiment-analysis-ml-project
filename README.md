# Sentiment Analysis MLOps Project

This project implements a sentiment analysis system using PyTorch Lightning,
with a focus on MLOps best practices.

## Project Overview

This is a machine learning project that uses deep learning for sentiment
analysis. The project is structured following MLOps principles, with clear
separation of concerns and reproducible training pipelines.

## Project Structure

```
├── Data/                   # Data directory (gitignored)
├── project/               # Main project code
│   └── my_modules/       # Core implementation modules
├── scripts/              # Utility and execution scripts
│   ├── data/            # Data processing scripts
│   └── download_data.py # Data download script
├── tests/               # Test files
├── training_logs/       # Training logs (gitignored)
└── lightning_logs/      # PyTorch Lightning logs (gitignored)
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

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd sentiment-analysis
```

2. Install Poetry (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:

```bash
poetry install
```

## Usage

1. Download the dataset:

```bash
poetry run python scripts/download_data.py
```

2. [Additional usage instructions will be added based on implementation details]

## Development

This project uses Poetry for dependency management. To add new dependencies:

```bash
poetry add package-name
```

## License

See the LICENSE file for details.

## Author

Irina Pavlova

---

**Note**: This is a work in progress. Documentation will be updated as the
project evolves.
