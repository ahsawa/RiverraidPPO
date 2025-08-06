import numpy as np


numbers = [4060, 6080, 4540, 4260, 6180]

media = np.mean(numbers)

std = np.std(numbers)

print(f"Mean: {media}")
print(f"Std:  {std:.2f}")