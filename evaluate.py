# evaluate.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(ga, best_weights, data):
    train_pred = ga.predict(data['X_train'], best_weights)
    test_pred = ga.predict(data['X_test'], best_weights)

    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*60)

    print("\nTập TRAIN:")
    print(f"  MAE : {mean_absolute_error(data['y_train'], train_pred):.4f} tỷ")
    print(f"  R²  : {r2_score(data['y_train'], train_pred):.4f}")

    print("\nTập TEST:")
    print(f"  MAE : {mean_absolute_error(data['y_test'], test_pred):.4f} tỷ")
    print(f"  R²  : {r2_score(data['y_test'], test_pred):.4f}")

    return test_pred