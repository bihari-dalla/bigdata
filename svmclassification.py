import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load iris data from Spark table
iris_pdf = spark.table("workspace.default.iris").toPandas()

# Prepare features and label
X = iris_pdf[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = iris_pdf["Species"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# Train SVM model
svm = SVC(class_weight='balanced')
svm.fit(Xtrain, ytrain)

# Predict and evaluate
predictions = svm.predict(Xtest)
print("Model Accuracy in testing = {}".format(accuracy_score(ytest, predictions)))

# After making predictions:
cm = confusion_matrix(ytest, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
