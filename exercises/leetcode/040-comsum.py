def bfs(l, target, prefix, result):
    if target < 0:
        return
    if target == 0:
        result.append(prefix)
        return
    if len(l) == 0:
        return

    bfs(l[1:], target - l[0], prefix + [l[0]], result)
    bfs(l[1:], target, prefix, result)


result = []
bfs(sorted([10, 1, 2, 7, 6, 1, 5]), 8, [], result)
print([list(t) for t in {tuple(l) for l in result}])
