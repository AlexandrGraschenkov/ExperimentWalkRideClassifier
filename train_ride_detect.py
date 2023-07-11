from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, ensemble
import os
from joblib import dump, load
from process_ride_detect import WalkDetect
prepare_data = WalkDetect.prepare_data


walk_dir = "/Users/alex/Downloads/walk_ride_data/walk"
ride_dir = "/Users/alex/Downloads/walk_ride_data/ride"
walk_files = [os.path.join(walk_dir, f) for f in os.listdir(walk_dir) if f.endswith(".csv")]
ride_files = [os.path.join(ride_dir, f) for f in os.listdir(ride_dir) if f.endswith(".csv")]

X_train = [prepare_data(f) for f in walk_files]
y_train = ["walk" for f in walk_files]
X_train += [prepare_data(f) for f in ride_files]
y_train += ["ride" for f in ride_files]

# model = DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Test the classifier on the test data and print the accuracy
print('Accuracy:', model.score(X_train, y_train))

trees = []
if isinstance(model, RandomForestClassifier):
    trees = model.estimators_
else:
    trees = [model]

for (i, t) in enumerate(trees):
    text_representation = tree.export_text(t)
    print(f"Tree: {i}:\n", text_representation)

dump(model, 'model.joblib') 

