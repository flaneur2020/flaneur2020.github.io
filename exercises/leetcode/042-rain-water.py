sample = [3, 1, 2, 5, 1, 4, 3]
lmax = [0] * len(sample)
rmax = [0] * len(sample)

lmax[0] = sample[0]
for i in range(1, len(sample)):
    if sample[i] > lmax[i-1]:
        lmax[i] = sample[i]
    else:
        lmax[i] = lmax[i-1]
print(lmax)

rmax[len(sample)-1] = sample[-1]
for i in range(len(sample)-2, -1, -1):
    rmax[i] = sample[i] if sample[i] > rmax[i+1] else rmax[i+1]
print(rmax)

result = 0
for i in range(1, len(sample)-1):
    count = min(lmax[i], rmax[i]) - sample[i]
    print(count)
    result += count

print(result)
