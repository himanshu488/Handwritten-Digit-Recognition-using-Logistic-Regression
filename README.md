# Handwritten Digits Recognition using Logistic Regression

# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

# Load the digits dataset
digits = load_digits()

# Print dataset description (optional)
# print(digits.DESCR)

# Visualize digits from index 23 to 27
for i in range(23, 28):
    plt.matshow(digits.images[i])
    plt.title(f"Label: {digits.target[i]}")
    plt.axis("off")
    plt.show()

# Features and labels
x = digits.data         # Each image as a 64-length feature vector
y = digits.target       # Corresponding labels (0-9)

# Split into training and test sets (80% training, 20% testing)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

# Initialize and train the model
model = LogisticRegression()
model.fit(train_x, train_y)

# Evaluate model accuracy
accuracy = model.score(test_x, test_y)
print(f"Model Accuracy: {accuracy:.2f}")

# Predict on a single test example
index = 190
predicted = model.predict(test_x[index: index + 1])[0]
actual = test_y[index]

print(f"Predicted: {predicted}, Actual: {actual}")

# Visualize the digit from the test set
plt.matshow(test_x[index].reshape(8, 8))
plt.title(f"Predicted: {predicted}, Actual: {actual}")
plt.axis("off")
plt.show()

