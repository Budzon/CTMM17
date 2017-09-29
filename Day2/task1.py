def longestCommonSubsequenceLength(p, q):
    lp = len(p)
    lq = len(q)
    commonSubsequenceLengthEndingAt = [[0 for k in range(lq + 1)] for n in range(lp + 1)]
    
    for i in range(lp):
        for j in range(lq):
            if (p[i] == q[j]):
                commonSubsequenceLengthEndingAt[i + 1][j + 1] = commonSubsequenceLengthEndingAt[i][j] + 1
            else:
                commonSubsequenceLengthEndingAt[i+1][j+1] = max(commonSubsequenceLengthEndingAt[i][j + 1], commonSubsequenceLengthEndingAt[i+1][j])
    return commonSubsequenceLengthEndingAt[-1][-1]