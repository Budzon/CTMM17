def positionToInsertInIncreasingSequence(p, x):
	if (len(p) == 0):
		return 0
	elif (len(p) == 1):
		return int(x > p[0])
	else:
		mid = len(p) / 2
		if (x < p[mid]):
			return positionToInsertInIncreasingSequence(p[0:mid], x)
		else:
			return positionToInsertInIncreasingSequence(p[mid:], x) + mid

def longestIncreasingSubsequence(p):
    if any(not (type(x) is int) for x in p):
        raise ValueError
    lp = len(p)
    # longestIncreasingSubsequenceEndingAt = [1 for i in range(lp)]
    
    # for i in range(lp - 1):
    # 	lengthBefore = 0
    # 	for j in range(i + 1):
    # 		if (p[j] < p[i + 1]):
    # 			lengthBefore = max(lengthBefore, longestIncreasingSubsequenceEndingAt[j])
    # 	longestIncreasingSubsequenceEndingAt[i + 1] = lengthBefore + 1

    # return longestIncreasingSubsequenceEndingAt
    maxP = max(p)
    lastNumberOfSubseqOfLen = [maxP + 1 for i in range(lp + 1)]
    lastNumberOfSubseqOfLen[0] = min(p) - 1
    for x in p:
    	lastNumberOfSubseqOfLen[positionToInsertInIncreasingSequence(lastNumberOfSubseqOfLen, x)] = x
    answer = 1
    while (answer < lp) and (lastNumberOfSubseqOfLen[answer + 1] < maxP + 1):
    	answer += 1
    return answer