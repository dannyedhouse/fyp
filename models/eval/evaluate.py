import matplotlib.pyplot as plt

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