
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """


def recursiveCombinationSum(candidates, target, memo):
    rs = set([])
    if memo.get(target) is not None:
        return memo[target]
    for c in candidates:
        n = target - c
        if n > 0:
            vs = recursiveCombinationSum(candidates, n, memo)
            for v in vs:
                rs.add(tuple(sorted(list(v) + [c])))
        elif n == 0:
            rs.add((c, ))
    print target, rs
    memo[target] = rs
    return rs


if __name__ == '__main__':
    memo = {}
    print recursiveCombinationSum([2, 3, 6, 7], 7, memo)
    memo = {}
    print recursiveCombinationSum([2, 3, 5], 8, memo)
