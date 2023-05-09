import random

arr = []
i = 0
while i < 9:
    arr.append(random.randint(0, 20))
    i = i + 1
print(arr)
arr.sort()
print('距离:', arr[4])
