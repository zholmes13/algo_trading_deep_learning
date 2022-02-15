import numpy as np

def calculate_weighted_f_score(y_true, y_pred):
    test_preds = np.argmax(y_pred, axis=1)
    ntu = sum((test_preds == 1) & (y_true == 1))
    Ntd = sum((test_preds == 0) & (y_true == 0))
    Ntf = sum((test_preds == 2) & (y_true == 2))
    Ewutd = sum((test_preds == 1) & (y_true == 0))
    Ewdtu = sum((test_preds == 0) & (y_true == 1))
    Ewutf = sum((test_preds == 1) & (y_true == 2))
    Ewdtf = sum((test_preds == 0) & (y_true == 2))
    Ewftu = sum((test_preds == 2) & (y_true == 1))
    Ewftd = sum((test_preds == 2) & (y_true == 0))

    beta_1 = 0.5
    beta_2 = 0.125
    beta_3 = 0.125

    Ntp = ntu + Ntd + beta_3**2 * Ntf
    E1 = Ewutd + Ewdtu
    E2 = Ewutf + Ewdtf
    E3 = Ewftu + Ewftd

    F = (1 + beta_1**2 + beta_2**2) * Ntp / ((1+beta_1**2+beta_2**2) * Ntp + E1 + beta_1**2 * E2 + beta_2**2 * E3)
    return F
