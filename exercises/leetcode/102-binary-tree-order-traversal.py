import collections


class Node(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


def dfs(n, depth, rs):
    rs.setdefault(depth, [])
    rs[depth].append(n.val)
    if n.left:
        dfs(n.left, depth+1, rs)
    if n.right:
        dfs(n.right, depth+1, rs)


def bfs(n):
    q = collections.deque()
    q.append(n)
    while len(q) > 0:
        n = q.popleft()
        print n.val
        if n.left:
            q.append(n.left)
        if n.right:
            q.append(n.right)


def levelOrder(n):
    rs = {}
    dfs(n, 0, rs)
    return [rs[i] for i in range(len(rs))]


if __name__ == '__main__':
    n = Node(3, Node(9), Node(20, Node(15), Node(7)))
    print levelOrder(n)
