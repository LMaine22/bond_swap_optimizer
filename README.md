# Bond Swap Optimizer

A professional bond portfolio optimization and analysis tool that helps identify optimal bond swap opportunities using genetic algorithms and multi-objective optimization.

## Features

- **Interactive Web Interface**: Modern Dash-based web application for easy portfolio analysis
- **Multiple Scenario Types**: 
  - Tax Loss Harvesting
  - Conservative Swaps
  - Yield Cleanup
  - Custom Configuration
- **Advanced Optimization**: NSGA-II genetic algorithm for multi-objective optimization
- **Comprehensive Analysis**: Risk-return profiles, recovery periods, and income enhancement metrics
- **Data Export**: Download results in CSV format for further analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bond_swap_optimizer.git
cd bond_swap_optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application
Run the interactive web application:
```bash
python dash_app.py
```
Then open your browser to `http://127.0.0.1:8050`

### Command Line
Run optimization from command line:
```bash
python main.py
```

## Data Format

The application expects bond portfolio data with the following required columns:
- `CUSIP`: Bond identifier
- `Current Face`: Par value
- `Book Price`: Book price (supports 32nds format)
- `Market Price`: Market price (supports 32nds format)

Optional columns:
- `Acctg Yield`: Accounting yield
- `Proj Yield (TE)`: Projected yield (tax equivalent)
- `Description`: Bond description

## Configuration

Edit `src/config.py` to customize:
- Scenario parameters
- Optimization settings
- Risk constraints
- Yield targets

## Project Structure

```
bond_swap_optimizer/
├── src/                    # Core optimization modules
│   ├── config.py          # Configuration and scenarios
│   ├── data_handler.py    # Data loading and preprocessing
│   ├── genetic_algorithm.py # NSGA-II optimization
│   ├── pruning.py         # Solution pruning and metrics
│   └── reporting.py       # Report generation
├── dash_app.py            # Web application
├── main.py               # Command-line interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- dash >= 2.14.0
- plotly >= 5.15.0
- geneticalgorithm >= 1.0.2
- scikit-learn >= 1.3.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or issues, please open an issue on GitHub.
