import numpy as np
import random

records = np.loadtxt('./ygyd_train.txt', dtype=object, delimiter=' ')
nums = [0] * 7
for r in records:
    nums[int(r[-1])] += 1
probs = [30000 / n / 10 for n in nums]
print(nums)
contents = []
for r in records:
    for i in range(10):
        if random.random() < probs[int(r[-1])]:
            contents.append(r)
nums = [0] * 7
for r in contents:
    nums[int(r[-1])] += 1
print(nums)
contents = np.stack(contents, 0)
np.savetxt('./augmented_data.txt', contents, fmt='%s', delimiter=' ')
