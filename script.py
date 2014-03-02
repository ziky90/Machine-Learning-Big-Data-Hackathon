import csv, time, numpy, neurolab, nltk
from sklearn import svm, grid_search, neighbors, preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords


"""
tweet object
"""
class Tweet:

	def __init__(self, date, label, text):
		self.date = time.strptime(date, "%y%m%d%H%M")
		self.label = label
		self.text = text


"""
reading tweets and saving them to the objects
"""

positiveText = []
negativeText = []
negList = []
posList = []
with open('hackathon-master/tweets/AAPL-train.csv', 'rb') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	for row in data:
		if(row[1] == '0.10'):
			positiveText.append(row[2])
			posList.append('positive')
			#positive.append(Tweet(row[0], row[1], row[2]))
		elif(row[1] == '-0.10'):
			negativeText.append(row[2])
			negList.append('negative')
			#negative.append(Tweet(row[0], row[1], row[2]))


postaged = zip(positiveText,posList)
negtaged = zip(negativeText,negList)
tagged = postaged + negtaged

tweets = []
for (word, sentiment) in tagged:
	word_filter = [i.lower() for i in word.split()]
	tweets.append((word_filter, sentiment))

def getwords(tweets):
	allwords = []
	for (words, sentiment) in tweets:
		allwords.extend(words)
	return allwords

def getwordfeatures(listoftweets):
	wordfreq = nltk.FreqDist(listoftweets)
	words = wordfreq.keys()
	return words

#print getwordfeatures(getwords(tweets))
#for tweet in tweets:
#	print tweet.text

wordlist = [i for i in getwordfeatures(getwords(tweets)) if not i in stopwords.words('english')]

def feature_extractor(doc):
	docwords = set(doc)
	features = {}
	for i in wordlist:
		features['contains(%s)' % i] = (i in docwords)
	return features

training_set = nltk.classify.apply_features(feature_extractor, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.show_most_informative_features(n=30)



"""
reading the training stocks data
"""
with open('hackathon-master/AAPL-train.csv', 'rb') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	indices = []
	matrix = []
	for row in data:
		if row[1] != 'close':
			list = []
			for i in range(len(row)-2):
				list.append(float(row[i+1]))
			label = float(row[7])
			if label > 0:
				indices.append(1)
			else:
				indices.append(0)
			matrix.append(list)
			#if label != 10 and label != -10:
			#	print label



"""
converting the loaded data from the list to the numpy array 
"""
X = numpy.array(matrix)
#X_scaled = preprocessing.scale(X)	#scaling the data for the SVM
#X = X[0:1000,:]			#using only first several samples to make it able of the computation
#print X_scaled
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)
print X_train_minmax
y = numpy.array(indices)
#y = y[0:1000]			#using only first several samples to make it able of the computation
print y
"""
training the SVM classifier and performing the cross validation to prevent 
the best success rate reached was around 49
algorithm is also too slow
"""
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 50, 100]}
#svr = svm.SVC()
#clf = grid_search.GridSearchCV(svr, parameters)
#clf.fit(X_scaled, y)
#clf = svm.SVC(kernel='rbf', C=1).fit(X_scaled, y)


"""
classification using knn
the best success rate reached was 51.12
probably linear time complexity
"""
clf1 = neighbors.KNeighborsClassifier(14, weights='distance')
clf1.fit(X, y)


"""
classification using adaboost
the best success rate reached was 49.0069
"""
#clf2 = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)


"""
Neural networks approach
Using Elman's recurent ANN
"""
#net = neurolab.net.newelm([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1]], [10, 1])
#error = net.train(X, y, epochs=500, show=100, goal=0.01)



"""
reading the dato to be classified
"""
with open('hackathon-master/AAPL-test.csv', 'rb') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	matrix = []
	close = []
	labels = []
	for row in data:
		if row[1] != 'close':
			list = []
			close.append(int(float(row[1])*1000))
			for i in range(len(row)-2):
				list.append(int(float(row[i+1])*1000))
			label = int(float(row[7])*100)
			labels.append(label)
			matrix.append(list)
toClassify = numpy.array(matrix)


"""
assigning the labels to the data based on the SVM classifier
"""
classes = clf1.predict(toClassify)  #for standard classifiers from sklearn library
#classes = net.sim(toClassify)		#for ANNs classifiers


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