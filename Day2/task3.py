def toPower(x, n):
	if (n == 0):
		return 1
	elif (n == 1):
		return x
	else:
		return toPower(x, n/2)**2 * toPower(x, n % 2)