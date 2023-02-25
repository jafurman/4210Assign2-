# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: naive_bayes.py
# SPECIFICATION: Output the classification of each test instance from the file weather_test (test set) if the classification confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: A while
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    read = csv.reader(csvfile)
    next(read)
    X = []
    Y = []
    for row in read:
        outlook = 1 if row[1] == 'Sunny' else 2 if row[1] == 'Overcast' else 3
        temperature = 1 if row[2] == 'Hot' else 2 if row[2] == 'Mild' else 3
        humidity = 1 if row[3] == 'High' else 2
        wind = 1 if row[4] == 'Weak' else 2
        # Append outlook, temp, humidity, and wind given conditions
        X.append([outlook, temperature, humidity, wind])
        Y.append(1 if row[5] == 'Yes' else 2)

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
# printing the header os the solution
# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        outlook = 1 if row[1] == 'Sunny' else 2 if row[1] == 'Overcast' else 3
        temperature = 1 if row[2] == 'Hot' else 2 if row[2] == 'Mild' else 3
        humidity = 1 if row[3] == 'High' else 2
        wind = 1 if row[4] == 'Weak' else 2
        prediction = clf.predict([[outlook, temperature, humidity, wind]])[0]
        probabilities = clf.predict_proba([[outlook, temperature, humidity, wind]])[0]
        confidence = probabilities[prediction - 1]
        play_tennis = 'Yes' if prediction == 1 else 'No'
        # Changing this variable will guarantee to show hte values of all those greater than or equal to the number.
        # The decimal is the success rate
        if confidence >= 0.75:
            print(row[0], row[1], row[2], row[3], row[4], play_tennis, confidence)
# I'm assuming this is what the problem was asking for, it'll print out the predicrtions with a value greater than
# .75 from the test sameple
