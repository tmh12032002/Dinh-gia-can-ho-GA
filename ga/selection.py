import random

def chon_loc_xac_suat(quan_the, danh_sach_fitness, so_luong_chon):
    tong_fitness = sum(danh_sach_fitness)
    xac_suat = [f / tong_fitness for f in danh_sach_fitness]

    cac_nst_duoc_chon = []
    for _ in range(so_luong_chon):
        r = random.random()
        tich_luy = 0
        for i, p in enumerate(xac_suat):
            tich_luy += p
            if r <= tich_luy:
                cac_nst_duoc_chon.append(quan_the[i])
                break
    return cac_nst_duoc_chon
