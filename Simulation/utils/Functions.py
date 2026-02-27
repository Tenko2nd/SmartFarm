import numpy as np

def add_realistic_noise(value, noise_level=0.01):
    """
    Adds Gaussian noise to sensor readings.
    Cannot go under 0
    """
    return max(0,value + np.random.normal(0, noise_level))