"""One normalization contract for original MNIST training and inference."""
import numpy as np


def prepare_mnist(images, invert=False):
    array = np.asarray(images, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3 or array.shape[1:] != (28, 28) or not len(array):
        raise ValueError('Expected a 28×28 grayscale image or a batch of them')
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
        raise ValueError('Pixels must be finite values between 0 and 255')
    if invert:
        array = 255 - array
    # Matches the original tf.keras.utils.normalize(batch, axis=1), including zeros.
    norms = np.sqrt(np.sum(array ** 2, axis=1, keepdims=True))
    return np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)
