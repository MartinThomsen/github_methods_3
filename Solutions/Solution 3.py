# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:48:48 2021

@author: rogue
"""

# Load data
import numpy as np
megmag = np.load('C:/Users/rogue/OneDrive/Dokumenter/MLSMML/github_methods_3/Solutions/Assignment 3/megmag_data.npy')
pas_vector = np.load('C:/Users/rogue/OneDrive/Dokumenter/MLSMML/github_methods_3/Solutions/Assignment 3/pas_vector.npy')

# Print info
print("Number of repetitions:", megmag.shape[0])
print("Number of sensors:", megmag.shape[1])
print("Number of time samples:", megmag.shape[2])

# Create list of times (indicating time in ms relative to onset of stimulus)
times = np.arange(-200, 804, 4).tolist()

import matplotlib.pyplot as plt

# Initialise matrix with zeroes
cov_matrix = np.zeros(shape = (102,102))
 
# Loop over each repetition and create covariance matrix
for i in range(682):
    cov_matrix += megmag[i,:,:] @ megmag[i,:,:].T
cov_matrix = cov_matrix/682

# Show figure and colour bar
plt.figure
plt.imshow(cov_matrix, cmap = 'hot')
plt.colorbar()
plt.show()

# Extract average
megmag_average = megmag.mean(axis = 0)

# Plot average magnetic field over time
plt.clf()
plt.plot(times, megmag_average.T)
plt.xlabel("Time (m/S)")
plt.ylabel("Average brain activity (Tesla)")
plt.title("Average brain activity over time")
plt.show()

# Identify and print maximal average magnetic field
peakMag_field = np.max(megmag_average)
print("Maximal average magnetic field:", peakMag_field, "Tesla")

# Extracting information on the index - this returns two values, the first identifies the sensor, the second the time point
index = np.unravel_index(np.argmax(megmag_average), shape = (102, 251))

# Printing information
print("Sensor with the maximal magnetic field is:", index[0])
print("The maximal magnetic field was measured at: ", times[index[1]], "ms")

#plt.clf()
#plt.figure()

# Plotting data for each repetition, at each time point, for sensor 73
plt.plot(times, megmag[:,73,:].T)
plt.axvline(x = times[112], color =  "black")
plt.xlabel("Time (ms)")
plt.ylabel("Brain activity measured at sensor 73 (Tesla)")
plt.title("All measurements from sensor 73")
plt.show()

pas_vector.shape

#plt.clf()
# Extracting data on sensor 73
sensor_data = megmag[:,73,:]

# Calculating average brain activity by pas ratings
pas1_avg = np.mean(sensor_data[np.where(pas_vector == 1)], axis = 0)
pas2_avg = np.mean(sensor_data[np.where(pas_vector == 2)], axis = 0)
pas3_avg = np.mean(sensor_data[np.where(pas_vector == 3)], axis = 0)
pas4_avg = np.mean(sensor_data[np.where(pas_vector == 4)], axis = 0)

# Plotting figure containing average brain activity for each pas rating at each time point
plt.figure()
plt.plot(times, pas1_avg)
plt.plot(times, pas2_avg)
plt.plot(times, pas3_avg)
plt.plot(times, pas4_avg)
plt.xlabel("Time (ms)")
plt.ylabel("Average brain activity (Tesla)")
plt.title("Average magnetic field for each pas-rating")
plt.legend(["pas 1", "pas 2", "pas 3", "pas 4"])
plt.show()

# Concatenate data - first all repetitions where pas score is 1 and 2
data_1_2 =np.concatenate((megmag[np.where(pas_vector == 1)], megmag[np.where(pas_vector == 2)]), axis = 0)

# Then as a vector of pas scores
y_1_2 = np.concatenate((pas_vector[np.where(pas_vector == 1)], pas_vector[np.where(pas_vector == 2)]), axis = 0)

X_1_2 = np.reshape(data_1_2, (214, 102*251))
X_1_2.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scaling X_1_2
X_scaled = scaler.fit_transform(X_1_2)

from sklearn.linear_model import LogisticRegression
# Creating a logistic regression with no penalty. The solver is not specified. Random state set to 42, because this is always the answer.
regr = LogisticRegression(random_state = 42, penalty = 'none')
regr.fit(X_scaled, y_1_2)

regr.score(X_scaled, y_1_2)

# Solver changed to liblinear, as default solver does not support l1 penalty. Tol set to 0.01 to avoid reaching maximum number of iterations. While this could also be changed to allow for more iterations, tol also sets a limit on desired accuracy
regr_penalty = LogisticRegression(random_state = 42, penalty = 'l1', solver = "liblinear")
regr_penalty.fit(X_scaled, y_1_2)
regr_penalty.score(X_scaled, y_1_2)

non_zero_coef = regr_penalty.coef_[regr_penalty.coef_ != 0]
print("Number of non-zero coefficients: ", len(non_zero_coef))

non_zero_indices = regr_penalty.coef_.flatten() != 0
X_reduced = X_scaled[:, non_zero_indices]

# Non-zero coefficients covariance matrix
cov_matrix = X_reduced @ np.transpose(X_reduced)

plt.close("all")
plt.figure()
plt.imshow(cov_matrix, cmap = 'hot')
plt.colorbar()
plt.show()

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Function to 
def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError("can't have more than two targets")
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
    # randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
    
    # create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y

X_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)
X_1_2_equal.shape
X_1_2_equal = X_1_2_equal.reshape(198, -1)
X_1_2_equal = scaler.fit_transform(X_1_2_equal)

cv = StratifiedKFold()
regr = LogisticRegression()
regr.fit(X_1_2_equal, y_1_2_equal)
scores = cross_val_score(regr, X_1_2_equal, y_1_2_equal, cv = 5)
mean_score = np.mean(scores)
print("Score across five folds: %.3f" % mean_score)

Cs = (1e5, 1e1, 1e-5)

for i in range(3):
    temp_regr = LogisticRegression(C = Cs[i], penalty = "l2");
    temp_regr.fit(X_1_2_equal, y_1_2_equal);
    scores[i] = np.mean(cross_val_score(temp_regr, X_1_2_equal, y_1_2_equal, cv = 5));
for i in range(3):
    print("Mean score utilising a Cs of %.5f: %.2f" % (Cs[i], scores[i]))
print("Best model is with a C = 1e-5, which is %.3f better than the initial fit" % (scores[2]-mean_score))

X_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)
cv = StratifiedKFold()
regr = LogisticRegression(C=1e-5, penalty="l2", solver = "liblinear")

scores = []

for i in range(251):
    t = scaler.fit_transform(X_1_2_equal[:,:,i]);
    regr.fit(t, y_1_2_equal);
    temp_scores = cross_val_score(regr, t, y_1_2_equal, cv=5);
    scores.append(np.mean(temp_scores))
    
X_1_2_equal.shape

print("Highest classification accuracy is: %.3f" % np.amax(scores))
print("The timepoint at which this occurs is: %.f" % np.argmax(scores))

plt.figure() 
plt.plot(times, scores)
plt.axvline(times[108], color = "black")
plt.axhline(y = 0.5, color = "green")
plt.xlabel("Time (ms)")
plt.ylabel("Classification accuracy")
plt.title("Classification accuracy at given times")
plt.show()

X_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)
cv = StratifiedKFold()
regr = LogisticRegression(C=1e-1, penalty="l1", solver = "liblinear")
scores = []

for i in range(251):
    t = scaler.fit_transform(X_1_2_equal[:,:,i]);
    regr.fit(t, y_1_2_equal);
    temp_scores = cross_val_score(regr, t, y_1_2_equal, cv=5);
    scores.append(np.mean(temp_scores))
    
print("Highest classification accuracy is: %.3f" % np.amax(scores))
print("The timepoint at which this occurs is: %.f" % np.argmax(scores))

plt.figure() 
plt.plot(times, scores)
plt.axvline(times[108], color = "black")
plt.axhline(y = 0.5, color = "green")
plt.xlabel("Time (ms)")
plt.ylabel("Classification accuracy")
plt.title("Classification accuracy at given times")
plt.show()
#%%
data_1_4 =np.concatenate((megmag[np.where(pas_vector == 1)], megmag[np.where(pas_vector == 4)]), axis = 0)
y_1_4 = np.concatenate((pas_vector[np.where(pas_vector == 1)], pas_vector[np.where(pas_vector == 4)]), axis = 0)

X_1_4_equal, y_1_4_equal = equalize_targets_binary(data_1_4, y_1_4)

cv = StratifiedKFold()
regr = LogisticRegression(C=1e-1, penalty="l1", solver = "liblinear")
scores = []

for i in range(251):
    t = scaler.fit_transform(X_1_4_equal[:,:,i]);
    regr.fit(t, y_1_4_equal);
    temp_scores = cross_val_score(regr, t, y_1_4_equal, cv=5);
    scores.append(np.mean(temp_scores))

print("Highest classification accuracy is: %.3f" % np.amax(scores))
print("The timepoint at which this occurs is: %.f" % np.argmax(scores))

plt.figure() 
plt.plot(times, scores)
plt.axvline(times[108], color = "black")
plt.axhline(y = 0.5, color = "green")
plt.xlabel("Time (ms)")
plt.ylabel("Classification accuracy")
plt.title("Classification accuracy at given times")
plt.show()

#%%
def equalize_targets(data, y):
    np.random.seed(7)
    targets = np.unique(y)
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target))
        indices.append(np.where(y == target)[0])
    min_count = np.min(counts)
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count, replace=False)
    third_choice = np.random.choice(indices[2], size=min_count, replace=False)
    fourth_choice = np.random.choice(indices[3], size=min_count, replace=False)
    
    new_indices = np.concatenate((first_choice, second_choice,
                                 third_choice, fourth_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y

megmag_equal, y_equal = equalize_targets(megmag, pas_vector)
megmag_equal.shape
y_equal.shape
megmag_equal = megmag_equal.reshape(396, -1)
megmag_equal.shape

from sklearn.svm import SVC
svm_linear = SVC(kernel="linear")
svm_radial = SVC(kernel="rbf")

megmag_scaled = scaler.fit_transform(megmag_equal)

scores_linear = cross_val_score(svm_linear, megmag_scaled, y_equal, cv=cv)
print("The linear classifier score is: %.3f", np.mean(scores_linear))


scores_radial = cross_val_score(svm_radial, megmag_scaled, y_equal, cv=cv)
print("The radial classifier score is: %.3f", np.mean(scores_radial))

megmag_equal, y_equal = equalize_targets(megmag, pas_vector)
cv = StratifiedKFold()
svm_radial = SVC(kernel="rbf")

scores = []

for i in range(251):
    t = scaler.fit_transform(megmag_equal[:,:,i]);
    regr.fit(t, y_equal);
    temp_scores = cross_val_score(svm_radial, t, y_equal, cv=5);
    scores.append(np.mean(temp_scores))

print("Highest classification accuracy is: %.3f" % np.amax(scores))
print("The timepoint at which this occurs is: %.f" % np.argmax(scores))

plt.figure() 
plt.plot(times, scores)
plt.axvline(times[226], color = "black")
plt.axhline(y = 0.25, color = "green")
plt.xlabel("Time (ms)")
plt.ylabel("Classification accuracy")
plt.title("Classification accuracy at given times")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(megmag_equal, y_equal, test_size = 0.30)

X_train = X_train.reshape(277, -1)
X_test = X_test.reshape(119, -1)
svm_radial.fit(X_train, y_train)
y_pred = svm_radial.predict(X_test)

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(svm_radial, X_test, y_test) # Plotting given estimator, test data and true labels.
plt.xlabel("Predicted PAS")
plt.ylabel("Observed PAS")
plt.title("Confusion Matrix PAS-by-PAS")
plt.show()