# FPL Squad Optimizer

Tools for extracting, predicting, and optimizing Fantasy Premier League (FPL) squads using machine learning. Includes dynamic formation selection, bench output, and full JSON export for frontend use.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Overview](#file-overview)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Data Extraction & Feature Engineering**: Reshapes and engineers FPL player data for ML prediction.
- **ML Prediction**: Uses Ridge and XGBoost (Optuna tuning) to predict next gameweek points for each player.
- **Squad Optimization**: Selects the optimal 15-player squad, starting XI, captain, vice-captain, and bench under FPL rules and budget constraints.
- **Transfer Suggestions**: Analyzes your current squad and recommends transfers to maximize predicted points (accounting for transfer costs).
- **Dynamic Formations**: Automatically selects the best formation (or user-specified) for the starting XI.
- **Bench Output**: Clearly separates starting XI and bench in both console and JSON output.
- **JSON Export**: Outputs full squad, starting XI, bench, captain, vice-captain, and formation to `optimal_squad.json` and `updated_squad.json` for frontend integration.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/elxecutor/fpl-squad.git
cd fpl-squad
pip install -r requirements.txt
```

## Usage
1. **Extract FPL player data**:
	```bash
	python export.py
	```
	This will fetch and flatten FPL player data from the API, saving it to `players.csv`.
2. **Run the full pipeline**:
	```bash
	python mvp_pipeline.py
	```
	This will reshape data, engineer features, and output `players_timeseries.csv`.
3. **Train the model and predict points**:
	```bash
	python train_model.py
	```
	This will train ML models and output predictions to `predicted_next_gw.csv`.
4. **Optimize squad and suggest transfers**:
	```bash
	python squad_optimizer.py
	# Or specify a formation:
	python squad_optimizer.py 3-5-2
	```
	This will load your current squad from `data/updated_squad.json`, check player availability, suggest transfers to maximize predicted points (accounting for 4-pt transfer hits), and output the updated squad, starting XI (sorted by position), bench, captain, vice-captain, and formation in both console and `updated_squad.json`. If no current squad exists, it optimizes a new squad and saves to `optimal_squad.json`.

## File Overview
- `export.py`: Extracts and flattens FPL player data from the API to `players.csv`.
- `mvp_pipeline.py`: Reshapes and engineers features for ML prediction, outputs `players_timeseries.csv`.
- `train_model.py`: Trains ML models and outputs predicted points to `predicted_next_gw.csv`.
- `squad_optimizer.py`: Loads predictions, optimizes new squad selection or analyzes current squad from `data/updated_squad.json` for transfer suggestions (accounting for 4-pt hits), outputs starting XI, bench, captain, vice-captain, and formation in both console and JSON files.
- `transfer_optimizer.py`: Analyzes a current squad (entered interactively) and suggests the best transfer to improve predicted points, outputs to console and `data/updated_squad.json`.
- `players.csv`: Output player data for ML and optimization.
- `players_timeseries.csv`: Feature-engineered player data for ML.
- `predicted_next_gw.csv`: Output predicted points for each player.
- `optimal_squad.json`: Full optimized squad, starting XI, bench, captain, vice-captain, and formation for frontend use (when no current squad is provided).
- `updated_squad.json`: Updated squad after applying transfer recommendations, including availability checks, transfer summary, and totals.
- `requirements.txt`: Python dependencies.

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or support, please open an issue or contact the maintainer via [X](https://www.x.com/elxecutor/).