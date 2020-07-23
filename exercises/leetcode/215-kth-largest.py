import heapq


def findKthLargest(nums, k):
    h = [None] * k
    heapq.heapify(h)
    for n in nums:
        heapq.heappush(h, n)
        heapq.heappop(h)
    return h[0]


print findKthLargest([-1, -1], 1)
