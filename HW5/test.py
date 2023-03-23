import random
import numpy as np

# generate a list with 100 random numbers
random_list = np.array(random.sample(range(0, 100), 100))

indexes = np.array(random.sample(range(0, 100), 5))

print(random_list)
print(indexes)
print(random_list[indexes])