# Apartment Valuation Project

This project implements a genetic algorithm for apartment valuation using machine learning techniques. The goal is to estimate the value of apartments based on various features and historical data.

## Project Structure

```
du_an_dinh_gia_can_ho
├── data
│   └── bo_du_lieu_can_ho.csv          # Dataset used for apartment valuation
├── ga
│   ├── __init__.py                     # Marks the ga directory as a Python package
│   ├── population.py                    # Manages a collection of individuals for the genetic algorithm
│   ├── fitness.py                       # Functions to calculate fitness of individuals
│   ├── selection.py                     # Implements selection methods for the genetic algorithm
│   ├── crossover.py                     # Defines crossover functions for producing offspring
│   ├── mutation.py                      # Contains mutation functions for introducing variability
│   └── run_ga.py                        # Main execution script for running the genetic algorithm
├── ml_model
│   ├── __init__.py                     # Marks the ml_model directory as a Python package
│   └── train_model.py                  # Functions for training a machine learning model
├── main.py                              # Entry point of the application
├── requirements.txt                     # Lists dependencies required for the project
└── README.md                            # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd du_an_dinh_gia_can_ho
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the project, execute the `main.py` file:
```
python main.py
```

## Modules Description

- **GA Module**: Implements the genetic algorithm components including population management, fitness evaluation, selection, crossover, and mutation.
- **ML Model Module**: Handles the training of machine learning models using the provided dataset, including preprocessing and evaluation.

## License

This project is licensed under the MIT License.