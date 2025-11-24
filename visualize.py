# visualize.py
import matplotlib.pyplot as plt

def plot_results(ga, data, test_pred, best_weights):
    plt.figure(figsize=(15, 5))

    # 1. Quá trình tiến hóa
    plt.subplot(1, 3, 1)
    plt.plot(ga.best_fitness_history, label='Best')
    plt.plot(ga.avg_fitness_history, label='Average', alpha=0.7)
    plt.title('Quá trình Tiến hóa')
    plt.xlabel('Thế hệ')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Dự đoán vs thực tế
    plt.subplot(1, 3, 2)
    plt.scatter(data['y_test'], test_pred, alpha=0.7, edgecolors='k')
    min_val = min(data['y_test'].min(), test_pred.min())
    max_val = max(data['y_test'].max(), test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.title('Dự đoán vs Thực tế')

    # 3. Sai số
    plt.subplot(1, 3, 3)
    errors = test_pred - data['y_test']
    plt.hist(errors, bins=10, edgecolor='black')
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Phân bố Sai số')
    plt.xlabel('Sai số (tỷ VNĐ)')

    plt.tight_layout()
    plt.savefig('ket_qua_dinh_gia_can_ho.png', dpi=300, bbox_inches='tight')
    plt.show()