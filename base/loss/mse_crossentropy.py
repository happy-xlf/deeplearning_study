import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    print(e_x)
    return e_x / e_x.sum(axis=0)

def mse(y_true, y_pred):
    """Mean Squared Error (MSE) loss function"""
    return np.mean(np.square(y_true - y_pred))

def cross_entropy(y_true, y_pred):
    """Cross-entropy loss function"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    print(y_pred)
    print(y_true * np.log(y_pred))
    return -np.sum(y_true * np.log(y_pred))


if __name__ == '__main__':
    y = np.asarray([1, 0, 0])
    logits = np.asarray([3,1,-3])
    print(softmax(logits))
    # print(mse(y, softmax(logits)))
    # print(cross_entropy(y, softmax(logits)))
    # cross_entropy_loss()