import random

def dot_bien(nst, xac_suat_dot_bien=0.05, sigma=0.05, lam_tron=4, clip=(-10, 10)):
    n = nst.copy()
    for i in range(len(n)):
        if random.random() < xac_suat_dot_bien:
            n[i] = round(n[i] + random.gauss(0, sigma), lam_tron)
            n[i] = max(clip[0], min(clip[1], n[i]))
    return n

def ap_dung_dot_bien(the_he_moi, xac_suat_dot_bien=0.05, sigma=0.05, lam_tron=4):
    return [dot_bien(nst, xac_suat_dot_bien, sigma, lam_tron) for nst in the_he_moi]
