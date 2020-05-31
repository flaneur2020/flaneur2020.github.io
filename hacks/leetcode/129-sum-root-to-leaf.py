class Node(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


def dfs(node, v, rs):
    v = v * 10 + node.val
    if not node.left and not node.right:
        rs.append(v)
    if node.left:
        dfs(node.left, v, rs)
    if node.right:
        dfs(node.right, v, rs)


if __name__ == '__main__':
    n = Node(4, Node(9, Node(5), Node(1)), Node(0))
    rs = []
    dfs(n, 0, rs)
    print rs
    print sum(rs)
