from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load the MNIST Data
mnist = load_digits()

x = mnist['data']  # (1797, 64)
y = mnist['target']  # (1797,)

# Create Training and Test Set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training a binary classifier to detect 5s
y_train_5 = y_train == 5
y_test_5 = y_test == 5

classer_5 = SGDClassifier(random_state=42)
classer_5.fit(x_train, y_train_5)

# Choose an image, make a prediction, and plot
some_digit = x[1001]
some_image = some_digit.reshape(8, 8)
value = y[1001]
plt.imshow(some_image, cmap='binary', interpolation='nearest')
plt.title(f"True Label: {value}")
plt.xlabel(f"Is the digit a 5: {classer_5.predict([some_digit])}")
plt.show()

# Cross validation
accuracy = cross_val_score(classer_5, x_train, y_train_5, cv=3, scoring='accuracy')
print(f"Accuracy of three cross-validation sets: {accuracy}")

# Confusion Matrix
y_train_pred = cross_val_predict(classer_5, x_train, y_train_5, cv=3)
cfm = confusion_matrix(y_train_5, y_train_pred)
print(f"Confusion Matrix: \n{cfm}")

# Precision and Recall
prec = precision_score(y_train_5, y_train_pred)
rec = recall_score(y_train_5, y_train_pred)
print(f"SGD Classifier\n\tPrecision Score: {prec}\n\tRecall Score: {rec}")

# F1 score
f1 = f1_score(y_train_5, y_train_pred)
print(f"\tF1 Score: {f1}")

# Visualize the threshold to balance the precision recall tradeoff
y_scores = cross_val_predict(classer_5, x_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Thresholds')
    plt.legend()
    plt.show()


plot_precision_recall_threshold(precisions, recalls, thresholds)


# Plot an ROC Curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


plot_roc_curve(fpr, tpr)
plt.title("SGD Classifier ROC Curve")
plt.show()

roc = roc_auc_score(y_train_5, y_scores)
print(f"SGD Classifier ROC AUC Score: {roc}")


# Lets train a Random Forest Classifier - this predicts probabilities
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method='predict_proba')
# we need scores though - use positive class prob as score
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# Plotting the comparison of the SGD Classifier and the Random Forest Classifier
plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.title("SGD vs Random Forest Classifier ROC Curve")
plt.legend()
plt.show()

froc = roc_auc_score(y_train_5, y_scores_forest)
print(f"Random Forest ROC AUC Score: {froc}")

y_forest_pred = cross_val_predict(forest_clf, x_train, y_train_5, cv=3)

prec = precision_score(y_train_5, y_forest_pred)
rec = recall_score(y_train_5, y_forest_pred)
print(f"Random Forest\n\tPrecision Score: {prec}\n\tRecall Score: {rec}")
