from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import dump
import os

def evaluate_classifiers(X_train, X_test, y_train, y_test, save_path=None):
    # === SVM ===
    print("=== SVM Report ===")
    svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(classification_report(y_test, y_pred_svm))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_svm)).plot(cmap='Blues')
    plt.title("Confusion Matrix - SVM")
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/conf_matrix_svm.png", dpi=300)
    plt.close()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dump(svm, save_path)

    # === KNN ===
    print("=== KNN Report ===")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(classification_report(y_test, y_pred_knn))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn)).plot(cmap='Oranges')
    plt.title("Confusion Matrix - KNN")
    plt.savefig("results/figures/conf_matrix_knn.png", dpi=300)
    plt.close()

    return svm  # opcional: se retorna el modelo entrenado
