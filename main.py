from os import listdir
import pandas as pd
import numpy as np
import entropy
# Machine learning tools From scikit-learn library.
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
# Classifiers used
from sklearn.tree import DecisionTreeClassifier


# Calculates different entropy values for a binary executable and creates a feature list.
def get_feature_list(content):
	v0 = entropy.shannon_entropy(content)
	v1 = sum(entropy.sample_entropy(content, 3))
	v2 = entropy.permutation_entropy(content)
	#v3 = entropy.weighted_permutation_entropy(content)
	#v4 = sum(entropy.multiscale_entropy(content, 3, maxscale=3))
	return [v0, v1, v2] # , v3, v4

# Create dataset by calculating entropy values for each file under './Packed' and './Not Packed' directories.
# This part is only run if we haven't created the dataset yet.
if "collection.csv" not in listdir():
	HEADERS = ["Shannon Entropy", "Sample Entropy", "Permutation Entropy", "Target"] #, "Weighted Permutation Entropy" , "Multiscale Entropy"
	entries = []

	for category, class_val in zip(["Not Packed", "Packed"], [0, 1]):
		for file in listdir(f"./{category}/"):
			print(f"Calculating values for './{category}/{file}'")
			with open(f"./{category}/{file}", "rb") as f:
				content = list(f.readline()) 
				# [v0, v1, v2, class_val]
				entries.append(get_feature_list(content)+[class_val])
				
	data_frame = pd.DataFrame(entries, columns=HEADERS)
	data_frame.to_csv("collection.csv", sep=",", header=True)

# If dataset is created and ready to use, Start training multiple classifiers for machine learning.
data_frame = pd.read_csv("collection.csv",  sep=",")
data_frame = data_frame.iloc[: , 1:] # Dropping the Index column as it confuses the learning algorithms.
data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
data_frame = data_frame.dropna(axis="index")

print("Number of entries in the dataframe: ", len(data_frame))
print("Class Distrubition")
print(data_frame["Target"].value_counts())

target_vals = data_frame["Target"]				# Only target column for every entry : y
entries = data_frame.drop(["Target"], axis=1)	# Every column except the target column for every entry : x
x_train, x_test, y_train, y_test = train_test_split(entries, target_vals, test_size=0.25, random_state=105) # Dividing the data into training and test sets with 75/25 ratio

classifier_object = DecisionTreeClassifier()
classifier_object.fit(x_train, y_train)
prediction = classifier_object.predict(x_test)
confusion = confusion_matrix(y_test, prediction) # Left diagonal values are the correct guesses and other values are mispredictions.
accuracy = accuracy_score(y_test, prediction) # Compares what model guessed and what it actually is to get the accuracy.
cvs = cross_val_score(classifier_object, x_test, y_test, cv=10).mean()
print("Classifier: {classifier} --> Accuracy: {accuracy:.4f}, Cross Validation Score: {cvs:.4f}".format(classifier=classifier_object, accuracy=accuracy, cvs=cvs))

# Confusion Matrix
print("   0   1")
for val,row in zip([0,1],confusion):
    print(val, row)


