
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import time


def accuracy(actual, guess):
    results = 0
    for i in range(len(actual)):
        if actual[i] == guess[i]:
            results += 1
    return results / float(len(actual))


def euclidean_distance(row1, row2):
    row1 = np.array(row1)
    row2 = np.array(row2)
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


'''
def cos_distance(row1, row2):
    x = np.array(row1)
    y = np.array(row2)
    cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return cos
'''


def normalize(data):
    minmax = []
    for i in range(len(data[0])):
        col = [row[i] for row in data]
        minmax.append([min(col), max(col)])
    for row in data:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])        #normalization formula


def k_nearest_neighbors(train, test, num_neighbors):
    predictions = []
    for row in test:

        distances = []
        for train_row in train:
            distance = euclidean_distance(row, train_row)           #calculate distance
            distances.append((train_row, distance))
        distances.sort(key=lambda x: x[1])                          #sort by the second item in the list
        # print(distances[1])
        # print(distances)
        neighbors = []
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])                       #add neighbors until k

        glass_type = [row[-1] for row in neighbors]                 #get the last column (labels)
        # print(glass_type)
        prediction = max(set(glass_type))                           #calculate which neighbors had the most votes

        predictions.append(prediction)
    return predictions


def cross_validation(data, folds):
    dataSplit = []
    tempData = list(data)
    fold_size = int(len(data) / folds)
    for i in range(folds):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(len(tempData))
            fold.append(tempData.pop(index))
        dataSplit.append(fold)
        #print(data_split)
    #print(data_split)
    return dataSplit


random.seed(1351351)

with open("glass.csv", "r", newline='') as file:
    reader = csv.reader(file)
    next(reader)
    data = []
    for row in reader:
        ri = float(row[0])
        Na = float(row[1])
        Mg = float(row[2])
        Al = float(row[3])
        Si = float(row[4])
        K = float(row[5])
        Ca = float(row[6])
        Ba = float(row[7])
        Fe = float(row[8])
        glass_type = int(row[9])

        data.append([ri, Na, Mg, Al, Si, K, Ca, Ba, Fe, glass_type])
    # print(tempData)
'''
with open("glass.csv", "r", newline='') as file:
    dict_reader = csv.DictReader(file)
    dictData = []
    for row in dict_reader:
        temp = dict(row)
        dictData.append(temp)

# print(dictData)
'''
fig, ax = plt.subplots(3, 3)

tempData = np.array(data)
print(tempData.shape)
ri = tempData[:, 0]
na = tempData[:, 1]
mg = tempData[:, 2]
al = tempData[:, 3]
si = tempData[:, 4]
k = tempData[:, 5]
ca = tempData[:, 6]
ba = tempData[:, 7]
fe = tempData[:, 8]
glass_t = tempData[:, 9]
ax[0, 0].scatter(glass_t, ri)
ax[0, 0].set_title("Refractive Index")
ax[0, 1].scatter(glass_t, na)
ax[0, 1].set_title("Sodium Content")
ax[0, 2].scatter(glass_t, mg)
ax[0, 2].set_title("Magnesium Content")
ax[1, 0].scatter(glass_t, al)
ax[1, 0].set_title("Aluminum Content")
ax[1, 1].scatter(glass_t, si)
ax[1, 1].set_title("Silicon Content")
ax[1, 2].scatter(glass_t, k)
ax[1, 2].set_title("Potassium Content")
ax[2, 0].scatter(glass_t, ca)
ax[2, 0].set_title("Calcium Content")
ax[2, 1].scatter(glass_t, ba)
ax[2, 1].set_title("Barium Content")
ax[2, 2].scatter(glass_t, fe)
ax[2, 2].set_title("Iron Content")
plt.show()

n_folds = 7
k = 5
means = []
normalize(data)
#print(data)
for n in range(1, k):
    start = time.time()
    folds = cross_validation(data, n_folds)
    #print(folds)
    scores = []
    x_axis = []
    y_axis = []
    for fold in folds:
        training = list(folds)
        training.remove(fold)
        training = sum(training, [])
        testing = []
        for row in fold:
            tempRow = list(row)
            testing.append(tempRow)
        results = k_nearest_neighbors(training, testing, n)
        actual = [row[-1] for row in fold]
        acc = accuracy(actual, results)
        #print(results)
        scores.append(acc)
    end = time.time()
    print('Scores: ', scores)
    mean = (sum(scores) / float(len(scores)) * 100)
    print('Mean accuracy of folds: ', mean, '%')
    means.append(mean)
    print('Time to execute: ', end - start, 'seconds')

print('Mean accuracy of different k values: ', sum(means) / float(len(means)), '%')
print('Average error rate: ', 100 - sum(means) / float(len(means)), '%')
print(means)
fig, ax = plt.subplots()
kBar = list(range(1, k))
ax.bar(kBar, means)
plt.xticks(kBar)
ax.set_xlabel('Fold #')
ax.set_ylabel('Accuracy %')
ax.set_title('Results')
plt.show()

