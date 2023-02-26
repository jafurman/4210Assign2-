# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: decision_tree_2
# SPECIFICATION: train, test, and output the performance of the 3 models created by using each training set on the test set provided (contact_lens_test.csv).
# FOR: CS 4210- Assignment #2
# TIME SPENT: 70 minutes
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    X_dictionary = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Myope': 1, 'Yes': 1, 'No': 2,
                    'Normal': 1, 'Reduced': 2, 'Astigmatism': 1, 'Hypermetrope': 2}
    for row in dbTraining:
        x_row = []
        for i in range(4):
            x_row.append(X_dictionary[row[i]])
        X.append(x_row)

    # transform the original categorical training classes to numbers and add to the vector Y.
    # For instance Yes = 1, No = 2
    for row in dbTraining:
        Y.append(X_dictionary[row[4]])

    avg_accuracy = 0
    # loop your training and test tasks 10 times here
    for i in range(10):
        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        dbTest = []
        # read the test data and add this data to dbTest
        test_file = 'contact_lens_test.csv'
        with open(test_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:
                    dbTest.append(row)

        correct_predictions = 0
        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during training,
            x_t = []
            for i in range(4):
                x_t.append(X_dictionary[data[i]])
            # and then use the decision tree to make the class prediction.
            class_predicted = clf.predict([x_t])[0]
            # compare the prediction with the true label
            # calculating the accuracy.
            if class_predicted == X_dictionary[data[4]]:
                correct_predictions += 1

        accuracy = correct_predictions / len(dbTest)
        avg_accuracy += accuracy

    avg_accuracy /= 10
    # print the average accuracy of this model during the 10 runs (training and test set).
    print(f"Final accuracy when training on {ds}: {avg_accuracy:.3f}")
