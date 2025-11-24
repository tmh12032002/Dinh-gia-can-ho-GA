# genetic_algorithm.py
import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=200, n_features=10, n_generations=500,
                 mutation_rate=0.1, crossover_rate=0.85):
        self.pop_size = pop_size
        self.n_features = n_features
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def initialize_population(self):
        return np.random.uniform(-1, 1, (self.pop_size, self.n_features + 1))

    def predict(self, X, weights):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X_with_bias, weights)

    def fitness_function(self, individual, X, y):
        pred = self.predict(X, individual)
        mae = np.mean(np.abs(y - pred))
        return 1.0 / (1.0 + mae)

    def selection(self, population, fitness_scores):
        selected = []
        for _ in range(self.pop_size):
            candidates = np.random.choice(len(population), 3, replace=False)
            winner = candidates[np.argmax(fitness_scores[candidates])]
            selected.append(population[winner].copy())
        return np.array(selected)

    def crossover(self, p1, p2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(p1))
            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])
            return c1, c2
        return p1.copy(), p2.copy()

    def mutation(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.5)
        return individual

    def evolve(self, X_train, y_train, patience=50, min_improvement=1e-5):
        population = self.initialize_population()
        best_individual = None
        best_fitness = 0
        no_improve = 0
        last_best = 0

        print("="*60)
        print("BẮT ĐẦU TIẾN HÓA")
        print("="*60)

        for gen in range(self.n_generations):
            fitnesses = np.array([self.fitness_function(ind, X_train, y_train) for ind in population])
            current_best = np.max(fitnesses)
            self.best_fitness_history.append(current_best)
            self.avg_fitness_history.append(np.mean(fitnesses))

            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmax(fitnesses)].copy()
                if current_best - last_best > min_improvement:
                    no_improve = 0
                    last_best = current_best
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if (gen + 1) % 10 == 0:
                mae = 1/best_fitness - 1
                print(f"Thế hệ {gen+1:3d} | Best Fitness: {best_fitness:.6f} | MAE: {mae:.4f} tỷ")

            if no_improve >= patience:
                print(f"\nEarly stopping tại thế hệ {gen+1}")
                break

            # Tiến hóa
            selected = self.selection(population, fitnesses)
            new_pop = []
            for i in range(0, self.pop_size, 2):
                p1 = selected[i]
                p2 = selected[(i+1) % self.pop_size]
                c1, c2 = self.crossover(p1, p2)
                new_pop.extend([self.mutation(c1), self.mutation(c2)])
            population = np.array(new_pop[:self.pop_size])

        print("HOÀN THÀNH TIẾN HÓA\n")
        return best_individual