# 10_CC.py
import numpy as np
import math
import random
import Queue

re_sample_acc = open("./09_BFS_MIT.txt", "r+") # input training data
line = re_sample_acc.readlines()
length = len(line)

element_init = line[0].split()


adj= []
for index in range(1, len(line), 1):
	element = line[index].split()
	adj.append([int(element[0]),int(element[1])])
	# if(int(element[0]) == 1):
	# 	print element

# print adj
# adj = [[4,6], [6,5], [4,3], [3,5], [2,1], [1,4] ]

frontier = []

color = []
distance = []
predecessor = []
for index in range(0,int(element_init[0])+1,1):
		# print index
		color.append('white')
		distance.append(-1)
		predecessor.append('Null')

# print color
# print distance
# print predecessor

color[1] = 'gray'
distance[1] = 0
predecessor[1] = 'Null'
q_BFS = Queue.Queue()
q_BFS.put(1)

# print color
# print distance
# print predecessor

while not q_BFS.empty():
	u = q_BFS.get()
	# print 'u=', u
	for index in range(0, len(adj), 1):
		if(adj[index][0] == u):
			v = adj[index][1]
			# print "v=", v
			if(color[v] == 'white'):
				color[v] = 'gray'
				distance[v] = distance[u] + 1
				predecessor[v] = u
				q_BFS.put(v)
				# print 'v=', v 
	color[u] = 'black'

# print color[1:]
# print len(distance)

for index in range (1, len(distance), 1):
	print distance[index], 

# print predecessor[1:]

# q = Queue.Queue()

# for i in range(5):
#     q.put(i)

# while not q.empty():
#     print q.get()