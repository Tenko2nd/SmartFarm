import numpy as np

def add_realistic_noise(value, noise_level=0.01):
    """
    Adds Gaussian noise to sensor readings.
    Cannot go under 0
    """
    return max(0,value + np.random.normal(0, noise_level))

def smootherstep(edge0, edge1, x):
    """
        Ken Perlin's SmootherStep.
        More 'biological' than SmoothStep, but stays in [0, 1].
    """
    x = max(0.0, min((x - edge0) / (edge1 - edge0), 1.0))
    # Evaluate polynomial: 6t^5 - 15t^4 + 10t^3
    return x**3 * (x * (x * 6 - 15) + 10)