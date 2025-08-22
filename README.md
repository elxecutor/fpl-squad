# FPL Squad Optimizer

Tools for extracting, predicting, and optimizing Fantasy Premier League (FPL) squads using machine learning and mathematical optimization.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Overview](#file-overview)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Data Extraction**: Fetches and flattens all FPL player data from the official API, including historical stats, and exports to `players.csv`.
- **Squad Optimization**: Loads player data, builds ML features, predicts next gameweek points, and selects the optimal 15-player squad and starting XI under FPL rules and budget constraints.
- **ML Prediction**: Uses XGBoost (if available) or RandomForest to predict expected points for each player.
- **Integer Linear Programming**: Uses PuLP to maximize squad points while respecting FPL constraints (budget, positions, max players per team).
- **Flexible Formation**: Automatically selects the best starting XI and formation.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/elxecutor/fpl-squad.git
cd fpl-squad
pip install -r requirements.txt
```

## Usage
1. **Export player data**:
	```bash
	python export.py
	```
	This will create `players.csv` with all player details.
2. **Run squad optimizer**:
	```bash
	python squad.py
	```
	This will output the optimized squad, starting XI, captain/vice, and save predictions to `predicted.csv`.

## File Overview
- `export.py`: Extracts and flattens FPL player data from the API to `players.csv`.
- `squad.py`: Loads `players.csv`, predicts points, and optimizes squad selection.
- `players.csv`: Output player data for ML and optimization.
- `predicted.csv`: Output predicted points for each player.
- `requirements.txt`: Python dependencies.

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or support, please open an issue or contact the maintainer via [LinkedIn](https://www.linkedin.com/in/iamgeekspe/).
