class Node(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


def dfs(node):
    if not node.right and not node.left:
        return True, node.val, node.val
    elif node.right and not node.left:
        ok, dmax, dmin = dfs(node.right)
        return (
            node.right.val > node.val and dmin > node.val and ok,
            max(node.val, dmax),
            min(node.val, dmin))
    elif node.left and not node.right:
        ok, dmax, dmin = dfs(node.left)
        return (
            node.left.val < node.val and dmax < node.val and ok,
            max(node.val, dmax),
            min(node.val, dmin))
    else:
        ok1, lmax, lmin = dfs(node.left)
        ok2, rmax, rmin = dfs(node.right)
        ok = node.left.val < node.val and \
            node.right.val > node.val and \
            ok1 and ok2 and \
            node.val > lmax and \
            node.val < rmin
        return ok, max(lmax, rmax), min(lmin, rmin)


if __name__ == '__main__':
    #    5
    #   /  \
    #  1    4
    #      /  \
    #     3    6
    n = Node(5, None, None)
    n.left = Node(1, None, None)
    n.right = Node(4, Node(3), Node(6))
    print dfs(n)

    n = Node(1, None, None)
    n.left = None
    n.right = Node(1)
    print dfs(n)

    #    3
    #  /    \
    # n       30
    #        /  \
    #      10   null
    #     /  \
    #    n    15
    #           \
    #            45
    n = Node(5, None, None)
    n.right = Node(30)
    n.right.left = Node(10)
    n.right.left.right = Node(15)
    n.right.left.right.right = Node(45)
    print dfs(n)

