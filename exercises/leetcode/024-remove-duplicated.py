def removeDuplicated(arr):
    if len(arr) == 0:
        return []
    j = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[j]:
            j += 1
            arr[j] = arr[i]
    return j + 1


if __name__ == '__main__':
    arr = [1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4]
    n = removeDuplicated(arr)
    print arr[:n]
