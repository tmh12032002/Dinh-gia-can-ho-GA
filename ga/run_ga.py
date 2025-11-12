import pandas as pd
import random
import copy
import math

from ga.population import khoi_tao_quan_the
from ga.fitness import tinh_danh_sach_fitness
from ga.selection import chon_loc_xac_suat
from ga.crossover import tao_the_he_lai
from ga.mutation import ap_dung_dot_bien
from ml_model.train_model import huan_luyen_mo_hinh, load_model_and_scaler

import pickle
import os

# đảm bảo model + scaler có sẵn
if not os.path.exists("ml_model.pkl"):
    huan_luyen_mo_hinh()

model, scaler = load_model_and_scaler("ml_model.pkl")
model_and_scaler = (model, scaler)

# Đọc dữ liệu
du_lieu = pd.read_csv("data/bo_du_lieu_can_ho.csv")

# Tham số GA (gợi ý)
NUM_FEATURES = du_lieu.shape[1] - 1
SO_THUOC_TINH = NUM_FEATURES + 1   # thêm 1 gene cho bias (intercept)
SO_THE_HE = 1000
SO_CA_THE = 100
SO_CHON = 30
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.06
ELITISM = True
PATIENCE = 25

# Khởi tạo quần thể ban đầu quanh hệ số của model nếu có
center = None
try:
    coef = list(model.coef_)
    intercept = float(model.intercept_)
    center = coef + [intercept]
except Exception:
    center = None

quan_the = khoi_tao_quan_the(SO_CA_THE, SO_THUOC_TINH, lam_tron=4, center=center, scale=0.1)

def tournament_selection(population, fitnesses, k=3):
    selected = []
    n = len(population)
    for _ in range(n):
        aspirants = random.sample(range(n), k)
        best = max(aspirants, key=lambda i: fitnesses[i])
        selected.append(copy.deepcopy(population[best]))
    return selected

best_overall = None
best_fitness = -math.inf
patience = 0

for gen in range(1, SO_THE_HE + 1):
    fitnesses = tinh_danh_sach_fitness(model_and_scaler, du_lieu, quan_the)
    gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    gen_best_fit = fitnesses[gen_best_idx]
    gen_best = quan_the[gen_best_idx]

    if gen_best_fit > best_fitness:
        best_fitness = gen_best_fit
        best_overall = copy.deepcopy(gen_best)
        patience = 0
    else:
        patience += 1

    avg_fit = sum(fitnesses) / len(fitnesses)
    print(f"Gen {gen:03d}: best_gen={gen_best_fit:.6f} | best_all={best_fitness:.6f} | avg={avg_fit:.6f} | patience={patience}")

    if patience >= PATIENCE and gen > 30:
        print("Early stopping vì không cải thiện.")
        break

    # Selection: nếu fitness hợp lệ dùng roulette, ngược lại tournament
    if sum(fitnesses) > 0 and not any(math.isnan(f) for f in fitnesses):
        parents = chon_loc_xac_suat(quan_the, fitnesses, SO_CHON)
    else:
        parents = tournament_selection(quan_the, fitnesses, k=3)

    offspring = tao_the_he_lai(parents)

    # ensure size
    while len(offspring) < SO_CA_THE:
        offspring.append(copy.deepcopy(random.choice(parents)))
    if len(offspring) > SO_CA_THE:
        offspring = offspring[:SO_CA_THE]

    # mutation (increase slightly if stuck)
    mut_rate = MUTATION_RATE if patience < 10 else min(0.15, MUTATION_RATE * 1.5)
    quan_the = ap_dung_dot_bien(offspring, xac_suat_dot_bien=mut_rate, sigma=0.05)

    # elitism: replace worst with best_overall
    if ELITISM and best_overall is not None:
        fits = tinh_danh_sach_fitness(model_and_scaler, du_lieu, quan_the)
        worst_idx = min(range(len(fits)), key=lambda i: fits[i])
        quan_the[worst_idx] = copy.deepcopy(best_overall)

# Kết quả
fitness_cuoi = tinh_danh_sach_fitness(model_and_scaler, du_lieu, quan_the)
best_index = fitness_cuoi.index(max(fitness_cuoi))
best_nst = quan_the[best_index]

print("\n🎯 Hệ số tốt nhất tìm được:", best_nst)
print("🔥 Fitness cao nhất:", max(fitness_cuoi))
# in MSE tương ứng
mse = (1.0 / max(fitness_cuoi) - 1.0) ** 2  # nghịch đảo của công thức fitness: fitness = 1/(1+sqrt(mse))
print(f"📉 MSE tương ứng (approx): {mse:.6f}")
