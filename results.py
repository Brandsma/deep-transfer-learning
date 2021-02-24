import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from logger import setup_logger

log = setup_logger(__name__)


def save_training_results(result_dir, name, history, config):
    log.info("Saving training results")

    # Save the history to a csv
    hist_df = pd.DataFrame.from_dict(history.history)
    hist_df.to_csv(result_dir + name + "-history.csv")

    # Show the training vs validation loss
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{config.model_type} loss")
    plt.legend()
    plt.savefig(result_dir + name + "-loss.png")

    # Show the training vs validation accuracy
    plt.figure()
    plt.plot(history.history["acc"], label="Training Accuracy")
    plt.plot(history.history["val_acc"], label="Validation Accuracy")
    plt.title(f"{config.model_type} accuracy")
    plt.legend()
    plt.savefig(result_dir + name + "-accuracy.png")


def save_test_results(result_dir, name, model, test_ds, class_names, config):
    test_results = model.evaluate(test_ds, verbose=1)
    with open(result_dir + name + "-test_results.txt", "w") as f:
        for element in test_results:
            f.write(str(element) + "\n")

    y_pred = model.predict(test_ds, verbose=1)
    y_true = tf.concat([y for x, y in test_ds], axis=0)
    predicted_categories = tf.argmax(y_pred, axis=1)
    fig = plot_confusion_matrix(
        y_true,
        predicted_categories,
        class_names,
        normalize=config.normalize_cm,
        title=f"{config.model_type} confusion matrix",
    )
    fig.savefig(result_dir + name + "confusion_matrix.png")

    predicted_label_batch = class_names[predicted_categories]

    # We take one batch of (16) images and show 9 of them
    # together with their predicted labels
    for image_batch, label_set in test_ds.take(1):
        plt.figure(figsize=(5, 5))
        plt.subplots_adjust(hspace=0.5)
        for n in range(min(9, config.batch_size)):
            plt.subplot(3, 3, n + 1)
            plt.imshow(image_batch[n])
            plt.title(predicted_label_batch[n].title())
            plt.axis("off")
            _ = plt.suptitle(f"{config.model_type} predictions")
            plt.savefig(result_dir + name + "-predictions.png")


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig
