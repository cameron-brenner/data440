#Cameron Brenner
#Project 00 - Problem 3 part b

import numpy
import scipy

AT = [[1, 2],
	 [4, -1],
	 [3, -3]]

B = [[-2, 0, 5],
     [0, -1, 4]]

	 
print(numpy.dot(AT, B))

print(numpy.linalg.matrix_rank(numpy.dot(AT, B)))