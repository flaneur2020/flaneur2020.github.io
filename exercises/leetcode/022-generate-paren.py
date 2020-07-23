def genp(n):
    return [
        ''.join(['(' if x == 1 else ')' for x in v])
        for v in recursiveGenP(n, n)]


def recursiveGenP(m, n):
    if m == 0 and n == 0:
        return [[]]
    rs = []
    if m > 0:
        for v in recursiveGenP(m-1, n):
            rs.append(v + [1])
    if n > 0:
        for v in recursiveGenP(m, n-1):
            if sum(v) - 1 >= 0:
                rs.append(v + [-1])
    return rs


if __name__ == '__main__':
    print genp(3)
