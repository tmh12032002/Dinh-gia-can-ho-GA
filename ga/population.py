import random

def khoi_tao_quan_the(so_luong, so_thuoc_tinh, lam_tron=4, center=None, scale=0.1):
    """
    Tạo quần thể gồm nhiều NST.
    Nếu center cung cấp (list), khởi tạo mỗi cá thể quanh center với Gaussian noise.
    so_thuoc_tinh: số gene (bao gồm bias ở cuối nếu dùng)
    """
    quan_the = []
    for _ in range(so_luong):
        if center is not None:
            # center có thể là list/array cùng chiều so_thuoc_tinh
            nst = [round(random.gauss(center[i], scale), lam_tron) for i in range(so_thuoc_tinh)]
        else:
            nst = [round(random.uniform(-1, 1), lam_tron) for _ in range(so_thuoc_tinh)]
        quan_the.append(nst)
    return quan_the
