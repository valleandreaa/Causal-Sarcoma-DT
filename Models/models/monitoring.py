from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

class PerformanceMonitor:
    def __init__(self):
        
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': []
        }

    def log_metrics(self, g_loss, d_loss, auc, precision, recall, f1, accuracy):
        self.history['g_loss'].append(g_loss)
        self.history['d_loss'].append(d_loss)
        self.history['auc'].append(auc)
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1'].append(f1)
        self.history['accuracy'].append(accuracy)

    def print_metrics(self, epoch):
        print(f"Epoch {epoch} - Generator Loss: {self.history['g_loss'][-1]:.4f}, Discriminator Loss: {self.history['d_loss'][-1]:.4f}")
        print(f"AUC-ROC: {self.history['auc'][-1]:.4f}, Precision: {self.history['precision'][-1]:.4f}, Recall: {self.history['recall'][-1]:.4f}")
        print(f"F1-Score: {self.history['f1'][-1]:.4f}, Accuracy: {self.history['accuracy'][-1]:.4f}")

    def plot_metrics(self):
        epochs = range(1, len(self.history['g_loss']) + 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.history['g_loss'], label='Generator Loss')
        plt.plot(epochs, self.history['d_loss'], label='Discriminator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history['auc'], label='AUC-ROC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC-ROC')

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history['precision'], label='Precision')
        plt.plot(epochs, self.history['recall'], label='Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Precision/Recall')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.history['f1'], label='F1-Score')
        plt.plot(epochs, self.history['accuracy'], label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('F1-Score/Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class GANMonitor:
    def __init__(self):
        self.epochs = []
        self.generator_losses = []
        self.discriminator_losses = []
        self.auc_scores = []
        self.real_labels = []
        self.fake_labels = []

    def update(self, epoch, g_loss, d_loss, real_labels, fake_labels):
        self.epochs.append(epoch)
        self.generator_losses.append(g_loss)
        self.discriminator_losses.append(d_loss)
        self.real_labels.append(real_labels)
        self.fake_labels.append(fake_labels)

        # Flatten lists of labels and predictions
        all_real_labels = np.concatenate(self.real_labels)
        all_fake_labels = np.concatenate(self.fake_labels)
        
        all_labels = np.concatenate([all_real_labels, all_fake_labels])
        all_predictions = np.concatenate([np.ones_like(all_real_labels), np.zeros_like(all_fake_labels)])
        
        if len(np.unique(all_labels)) > 1:  # Check if there are both real and fake samples
            auc_score = roc_auc_score(all_labels, all_predictions)
            self.auc_scores.append(auc_score)
        else:
            self.auc_scores.append(None)  # In case of no variance in labels

    def plot_metrics(self):
        # Plot losses
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.generator_losses, label='Generator Loss')
        plt.plot(self.epochs, self.discriminator_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses')

        # Plot AUC-ROC
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.auc_scores, label='AUC-ROC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC-ROC')
        plt.legend()
        plt.title('AUC-ROC')

        plt.tight_layout()
        plt.show()