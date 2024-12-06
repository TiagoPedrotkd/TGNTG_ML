import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].legend()
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].legend()
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_val, y_val, class_mapping, cmap='Blues', title="Matriz de Confusão - Conjunto de Validação"):
    
    y_val_preds = np.argmax(model.predict(X_val), axis=1)

    conf_matrix = confusion_matrix(y_val, y_val_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(class_mapping.values()))
    disp.plot(cmap=cmap, xticks_rotation='vertical')

    plt.title(title)
    plt.show()

    report = classification_report(y_val, y_val_preds, target_names=list(class_mapping.values()))
    print(report)
