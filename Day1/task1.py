def countDistinct(arr, maxVal):
    counter = [0] * maxVal
    numberOfDistinct = 0

    for x in arr:
        if (counter[x-1] == 0):
            numberOfDistinct += 1
        counter[x-1] += 1
    return numberOfDistinct