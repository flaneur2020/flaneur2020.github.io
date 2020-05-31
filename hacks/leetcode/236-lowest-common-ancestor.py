class Node(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


def dfs(node, p, q):
    hasq1, hasq2, hasp1, hasp2 = False, False, False, False
    n1, n2 = None, None
    if node.left:
        hasp1, hasq1, n1 = dfs(node.left, p, q)
    if node.right:
        hasp2, hasq2, n2 = dfs(node.right, p, q)
    hasp = hasp1 or hasp2 or node.val == p
    hasq = hasq1 or hasq2 or node.val == q
    n = node if hasp and hasq and (not n1 and not n2) else (n1 or n2)
    return (hasp, hasq, n)


if __name__ == '__main__':
    node = Node(
        3,
        Node(5, Node(6), Node(2, Node(7), Node(4))),
        Node(1, Node(0), Node(8)))
    _, _, n = dfs(node, 5, 4)
    print n.val
