# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: knn.py. even though I called this main.py
# SPECIFICATION: read the file binary_points.csv and output the LOO-CV error rate for 1NN (same answer as part a).
# FOR: CS 4210- Assignment #2
# TIME SPENT: a while
# -----------------------------------------------------------*/

from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# loop your data to allow each instance to be your test set
error_count = 0
for i in range(len(db)):

    # add the training features to the 2D array X removing the instance that will be used for testing in this iteration
    X = []
    for j in range(len(db)):
        if i != j:
            X.append([float(db[j][0]), float(db[j][1])])

    # transform the original training classes to numbers and add to the vector Y removing the instance that will be
    # used for testing in this iteration
    Y = []
    for j in range(len(db)):
        if i != j:
            if db[j][2] == '+':
                Y.append(1)
            else:
                Y.append(0)

    # store the test sample of this iteration in the vector testSample
    testSample = [float(db[i][0]), float(db[i][1])]

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction
    class_predicted = clf.predict([testSample])[0]

    # compare the prediction with the true label of the test instance to start calculating the error rate
    if db[i][2] == '+' and class_predicted == 0:
        error_count += 1
    elif db[i][2] == '-' and class_predicted == 1:
        error_count += 1

# print the error rate
print("LOO-CV Error Rate:", error_count / len(db))