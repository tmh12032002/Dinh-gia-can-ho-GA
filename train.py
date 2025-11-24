# train.py
from genetic_algorithm import GeneticAlgorithm
from data import load_and_prepare_data

def train_model():
    data = load_and_prepare_data()
    ga = GeneticAlgorithm(
        pop_size=200,
        n_features=data['X_train'].shape[1],
        n_generations=500,
        mutation_rate=0.1,
        crossover_rate=0.85
    )

    print("Đang huấn luyện mô hình bằng Giải thuật Di truyền...")
    best_weights = ga.evolve(data['X_train'], data['y_train'], patience=50)
    
    return ga, best_weights, data