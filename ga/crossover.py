import random

def lai_mot_diem(nst1, nst2):
    do_dai = len(nst1)
    diem = random.randint(1, do_dai - 1)
    con1 = nst1[:diem] + nst2[diem:]
    con2 = nst2[:diem] + nst1[diem:]
    return con1, con2

def tao_the_he_lai(cac_nst_duoc_chon):
    the_he_moi = []
    for i in range(0, len(cac_nst_duoc_chon) - 1, 2):
        cha1 = cac_nst_duoc_chon[i]
        cha2 = cac_nst_duoc_chon[i + 1]
        con1, con2 = lai_mot_diem(cha1, cha2)
        the_he_moi.extend([con1, con2])
    if len(cac_nst_duoc_chon) % 2 == 1:
        the_he_moi.append(cac_nst_duoc_chon[-1])
    return the_he_moi
