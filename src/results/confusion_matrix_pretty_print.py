import seaborn as sns
import matplotlib.pyplot as plt


def print_confusion_matrix(cm):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax);  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['business', 'health']);
    ax.yaxis.set_ticklabels(['health', 'business'])
    plt.show()
