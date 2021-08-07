import copy


def dfs(nums, prefix, result):
    if len(nums) == 0:
        result.append(prefix)
    for i in range(len(nums)):
        nums[i], nums[0] = nums[0], nums[i]
        dfs(copy.copy(nums[1:]), prefix + [nums[i]], result)


result = []
dfs([1, 2, 3], [], result)
print(result)
