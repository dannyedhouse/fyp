import matplotlib.pyplot as plt
import itertools
import numpy as np

def display_accuracy_graph(history):
    """Plots a graph with the accuracy and val_accuracy to help determine best number of epochs"""
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()

def display_loss_graph(history):
    """Plots a graph with the loss and val_loss"""
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'val_loss'])
    plt.show()

def create_confusion_matrix(confusion_matrix, classes):
    """Plots confusion matrix
    ~ based on https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - BBC Categorization", fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")