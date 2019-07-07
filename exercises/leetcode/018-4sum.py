def recursiveFourSum(nums, target):
    numIdxMap = {nums[i]: i for i in range(0, len(nums))}
    rs = set([])
    for a in range(0, len(nums)-3):
        for b in range(a+1, len(nums)-2):
            for c in range(b+1, len(nums)-1):
                n = target - nums[a] - nums[b] - nums[c]
                idx = numIdxMap.get(n)
                if idx is not None and idx > c:
                    rs.add((nums[a], nums[b], nums[c], n))
    return rs


if __name__ == '__main__':
    print list(recursiveFourSum(sorted([1, 0, -1, 0, -2, 2]), 0))
