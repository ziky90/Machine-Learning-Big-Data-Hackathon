import datetime, csv, numpy
from sklearn.hmm import GaussianHMM
"""
Stock price prediction using only volume and labels for the HMM model. Chosen size 2 and one state 1 represents buy, state 2 sell
"""

"""
reading the training stocks data
"""
with open('hackathon-master/AAPL-train.csv', 'rb') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	indices = []
	volume = []
	for row in data:
		if row[1] != 'close':
			#list = []
			#for i in range(len(row)-2):
			#	list.append(float(row[i+1]))
			label = float(row[7])
			volume.append(float(row[2]))
			if label > 0:
				indices.append(1)
			else:
				indices.append(0)
			#matrix.append(list)

X = numpy.column_stack([numpy.array(indices), numpy.array(volume)])
model = GaussianHMM(2, covariance_type="diag", n_iter=1000)

model.fit([X])


"""
reading the dato to be classified
"""
with open('hackathon-master/AAPL-test.csv', 'rb') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	#matrix = []
	volume = []
	labels = []
	for row in data:
		if row[1] != 'close':
			list = []
			volume.append(float(row[2]))
			#for i in range(len(row)-2):
			#	list.append(int(float(row[i+1])*1000))
			label = int(float(row[7])*100)
			labels.append(label)
			#matrix.append(list)

"""
Building the HMM
"""
X = numpy.column_stack([numpy.array(labels), numpy.array(volume)])
classes = model.predict(X)

"""
calculating the algorithm performance
"""
result = 0
correct = 0
correctTens = 0
totalTens = 0
trades = 0
for i in range(len(classes)):
	print classes[i], labels[i]
	if (classes[i] == 0 and labels[i] == -10) or (classes[i] == 1 and labels[i] == 10):
		correct += 1
	if classes[i] == 1 or classes[i] == 0 or labels == 1 or labels == 0:
		totalTens += 1
		if (classes[i] == 0 and labels[i] == -10) or (classes[i] == 1 and labels[i] == 10):
			correctTens += 1

	if(classes[i] == 1):
		result += labels[i]
		trades += 1

"""
printing the performance numbers
"""  
print result	#sum of all the labels where I have decided for buy

print float(correct*100)/float(len(labels))	#correctly predicted ration

print float(correctTens*100)/float(totalTens)	#correctly predicted ratio dealing only with tens		

print trades 	#number of total trades