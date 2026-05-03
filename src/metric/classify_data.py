import numpy as np
from src.activation.activation import Array


def classify_data (zeta: Array, output: Array,threshold:float =0.8)->Array:

    zeta = np.asarray(zeta).reshape(-1)
    output = np.asarray(output).reshape(-1)

    arr_output_size = output.shape[0]
    arr_zeta_size = zeta.shape[0]
    if arr_output_size != arr_zeta_size:
        raise ValueError("zeta and output have different shapes")

    false_pos, false_neg, true_pos, true_neg = np.zeros(4)
    for index in range(arr_output_size):
        adjusted_output = 1 if output[index] >= threshold else 0
        if zeta[index] == 1 and adjusted_output == 1:
            true_pos += 1
        elif zeta[index] == 0 and adjusted_output == 1:
            false_pos += 1
        elif zeta[index] == 1 and adjusted_output == 0:
            false_neg += 1
        elif zeta[index] == 0 and adjusted_output == 0:
            true_neg += 1

    return np.array([false_pos, false_neg, true_pos, true_neg])
