# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        return addTwo(l1, l2, 0)


def addTwo(l1, l2, inc):
    if not l1 and not l2:
        if inc:
            return ListNode(inc)
        return None
    v1 = l1.val if l1 else 0
    v2 = l2.val if l2 else 0
    v = v1 + v2 + inc
    next1 = l1.next if l1 else None
    next2 = l2.next if l2 else None
    n = ListNode(v % 10)
    n.next = addTwo(next1, next2, int(v >= 10))
    return n


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


if __name__ == '__main__':
    n1 = ListNode(1)
    n1.next = ListNode(2)
    n1.next.next = ListNode(3)
    n2 = ListNode(1)
    n2.next = ListNode(2)
    n2.next.next = ListNode(3)
    nr = addTwo(n1, n2, 0)
    print nr.val
    print nr.next.val
    print nr.next.next.val
    print nr.next.next.next
