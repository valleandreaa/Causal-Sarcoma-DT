import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

class GANLossMonitor:
    def __init__(self):
        self.generator_losses = []
        self.discriminator_losses = []
        self.discriminator_accuracies = []
        self.generator_diversity = []  # Mode collapse detection
        self.prediction_errors = []  # For tracking prediction error (e.g., MSE)
        self.inception_scores = []  # Inception Score tracking
        self.epochs = []
        self.real_discriminator_error = []

        # Initialize the plot
        self.fig, self.ax = plt.subplots(6, 1, figsize=(10, 25))


        self.gen_line, = self.ax[0].plot([], [], label='Generator Loss')
        self.disc_line, = self.ax[0].plot([], [], label='Discriminator Loss')
        self.ax[0].set_xlabel('Epoch')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].legend()
        self.ax[0].set_title('Losses During Training')

        self.disc_acc_line, = self.ax[1].plot([], [], label='Discriminator Accuracy')
        self.ax[1].set_xlabel('Epoch')
        self.ax[1].set_ylabel('Accuracy')
        self.ax[1].legend()
        self.ax[1].set_title('Discriminator Accuracy')

        self.pred_error_line, = self.ax[2].plot([], [], label='Prediction Error')
        self.ax[2].set_xlabel('Epoch')
        self.ax[2].set_ylabel('Error')
        self.ax[2].legend()
        self.ax[2].set_title('Prediction Error (e.g., MSE)')

        self.inception_score_line, = self.ax[3].plot([], [], label='Inception Score')
        self.ax[3].set_xlabel('Epoch')
        self.ax[3].set_ylabel('Score')
        self.ax[3].legend()
        self.ax[3].set_title('Inception Score')

        self.diversity_line, = self.ax[4].plot([], [], label='Diversity Score')  # New line for diversity score
        self.ax[4].set_xlabel('Epoch')
        self.ax[4].set_ylabel('Score')
        self.ax[4].legend()
        self.ax[4].set_title('Generator Diversity Score')

        self.real_discriminator_line, = self.ax[5].plot([], [], label='Discriminator Error Real')  # New line for diversity score
        self.ax[5].set_xlabel('Epoch')
        self.ax[5].set_ylabel('Error')
        self.ax[5].legend()
        self.ax[5].set_title('Discriminator Error Real')

    def update(self, epoch, gen_loss, disc_loss, disc_acc=None, pred_error=None, inception_score=None, gen_div=None, disc_error=None):
        """
        Update the loss lists and plot the losses.
        
        Parameters:
        epoch (int): The current epoch number.
        gen_loss (float): The generator loss.
        disc_loss (float): The discriminator loss.
        disc_acc (float): The discriminator accuracy (optional).
        pred_error (float): The prediction error, like MSE (optional).
        inception_score (float): The inception score (optional).
        gen_div (float): The generator diversity score (optional).
        """
        self.epochs.append(epoch)
        self.generator_losses.append(gen_loss)
        self.discriminator_losses.append(disc_loss)
        
        if disc_acc is not None:
            self.discriminator_accuracies.append(disc_acc)
        if pred_error is not None:
            self.prediction_errors.append(pred_error)
        if inception_score is not None:
            self.inception_scores.append(inception_score)
        if gen_div is not None:
            self.generator_diversity.append(gen_div)
        
        if disc_error is not None:
            self.real_discriminator_error.append(disc_error)

        self.plot_losses()

    def plot_losses(self):
        """
        Plot the generator and discriminator losses, and other metrics.
        """
        clear_output(wait=True)  # Keep this to clear previous plots for smoother updating

        # Plot the generator and discriminator losses
        self.gen_line.set_data(self.epochs, self.generator_losses)
        self.disc_line.set_data(self.epochs, self.discriminator_losses)
        self.ax[0].relim()  # Recompute the limits
        self.ax[0].autoscale_view()  # Auto-scale the view to the data

        # Plot discriminator accuracy
        if self.discriminator_accuracies:
            self.disc_acc_line.set_data(self.epochs, self.discriminator_accuracies)
            self.ax[1].relim()
            self.ax[1].autoscale_view()

        # Plot prediction error
        if self.prediction_errors:
            self.pred_error_line.set_data(self.epochs, self.prediction_errors)
            self.ax[2].relim()
            self.ax[2].autoscale_view()

        # Plot Inception Score
        if self.inception_scores:
            self.inception_score_line.set_data(self.epochs, self.inception_scores)
            self.ax[3].relim()
            self.ax[3].autoscale_view()

        # Plot Generator Diversity Score
        if self.generator_diversity:
            self.diversity_line.set_data(self.epochs, self.generator_diversity)
            self.ax[4].relim()
            self.ax[4].autoscale_view()

        if self.real_discriminator_error:
            self.real_discriminator_line.set_data(self.epochs, self.real_discriminator_error)
            self.ax[5].relim()
            self.ax[5].autoscale_view()

        self.fig.canvas.draw()  # Draw the updated figure
        self.fig.canvas.flush_events()  # Ensure the update is displayed
        plt.pause(0.001)  # Pause to allow for display refresh




class CycleGANLossMonitor:
    """Loss monitor tailored for CycleGAN training."""

    def __init__(self, path=None):
        self.generator_losses = []
        self.discriminator_losses = []
        self.cycle_losses = []
        self.identity_losses = []
        self.gx_losses = []
        self.gy_losses = []
        self.dx_losses = []
        self.dy_losses = []
        self.val_generator_losses = []
        self.val_discriminator_losses = []
        self.val_cycle_losses = []
        self.val_identity_losses = []
        self.val_gx_losses = []
        self.val_gy_losses = []
        self.val_dx_losses = []
        self.val_dy_losses = []
        self.epochs = []
        self.path = path

        self.fig, self.ax = plt.subplots(5, 1, figsize=(10, 25))


        # Subplot 0: Generator and Discriminator Losses (Train & Val)
        self.gen_line, = self.ax[0].plot([], [], label='Train Generator Loss')
        self.gen_val_line, = self.ax[0].plot([], [], label='Val Generator Loss')
        self.disc_line, = self.ax[0].plot([], [], label='Train Discriminator Loss')
        self.disc_val_line, = self.ax[0].plot([], [], label='Val Discriminator Loss')
        self.ax[0].set_xlabel('Epoch')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].legend()
        self.ax[0].set_title('Overall Losses')

        # Subplot 1: Generator component losses (Gx & Gy)
        self.gx_line, = self.ax[1].plot([], [], label='Train Gx Loss')
        self.gy_line, = self.ax[1].plot([], [], label='Train Gy Loss')
        self.gx_val_line, = self.ax[1].plot([], [], label='Val Gx Loss')
        self.gy_val_line, = self.ax[1].plot([], [], label='Val Gy Loss')
        self.ax[1].set_xlabel('Epoch')
        self.ax[1].set_ylabel('Loss')
        self.ax[1].legend()
        self.ax[1].set_title('Generator Losses')

        # Subplot 2: Discriminator component losses (Dx & Dy)
        self.dx_line, = self.ax[2].plot([], [], label='Train Dx Loss')
        self.dy_line, = self.ax[2].plot([], [], label='Train Dy Loss')
        self.dx_val_line, = self.ax[2].plot([], [], label='Val Dx Loss')
        self.dy_val_line, = self.ax[2].plot([], [], label='Val Dy Loss')
        self.ax[2].set_xlabel('Epoch')
        self.ax[2].set_ylabel('Loss')
        self.ax[2].legend()
        self.ax[2].set_title('Discriminator Losses')

        # Subplot 3: Cycle Consistency Losses (Train & Val)
        self.cycle_loss_line, = self.ax[3].plot([], [], label='Train Cycle Loss')
        self.cycle_loss_val_line, = self.ax[3].plot([], [], label='Val Cycle Loss')
        self.ax[3].set_xlabel('Epoch')
        self.ax[3].set_ylabel('Loss')
        self.ax[3].legend()
        self.ax[3].set_title('Cycle Consistency Loss')

        # Subplot 4: Identity Losses (Train & Val)
        self.identity_loss_line, = self.ax[4].plot([], [], label='Train Identity Loss')
        self.identity_loss_val_line, = self.ax[4].plot([], [], label='Val Identity Loss')
        self.ax[4].set_xlabel('Epoch')
        self.ax[4].set_ylabel('Loss')
        self.ax[4].legend()
        self.ax[4].set_title('Identity Loss')

    def update(
        self,
        epoch,
        gen_loss,
        disc_loss,
        cycle_loss=None,
        identity_loss=None,
        val_gen_loss=None,
        val_disc_loss=None,
        val_cycle_loss=None,
        val_identity_loss=None,
        gx_loss=None,
        gy_loss=None,
        dx_loss=None,
        dy_loss=None,
        val_gx_loss=None,
        val_gy_loss=None,
        val_dx_loss=None,
        val_dy_loss=None,
        **kwargs,
    ):
        """Update stored metrics and refresh the plots."""
        self.epochs.append(epoch)
        self.generator_losses.append(gen_loss)
        self.discriminator_losses.append(disc_loss)

        if cycle_loss is not None:
            self.cycle_losses.append(cycle_loss)
        if identity_loss is not None:
            self.identity_losses.append(identity_loss)
        if val_gen_loss is not None:
            self.val_generator_losses.append(val_gen_loss)
        if val_disc_loss is not None:
            self.val_discriminator_losses.append(val_disc_loss)
        if val_cycle_loss is not None:
            self.val_cycle_losses.append(val_cycle_loss)
        if val_identity_loss is not None:
            self.val_identity_losses.append(val_identity_loss)

        if gx_loss is not None:
            self.gx_losses.append(gx_loss)
        if gy_loss is not None:
            self.gy_losses.append(gy_loss)
        if dx_loss is not None:
            self.dx_losses.append(dx_loss)
        if dy_loss is not None:
            self.dy_losses.append(dy_loss)
        if val_gx_loss is not None:
            self.val_gx_losses.append(val_gx_loss)
        if val_gy_loss is not None:
            self.val_gy_losses.append(val_gy_loss)
        if val_dx_loss is not None:
            self.val_dx_losses.append(val_dx_loss)
        if val_dy_loss is not None:
            self.val_dy_losses.append(val_dy_loss)

        self.plot_losses()
        # Save updated plot to a single file
        if self.path:
            self.fig.savefig(self.path)

    def plot_losses(self):
        clear_output(wait=True)

        # Subplot 0: Generator and Discriminator losses
        self.gen_line.set_data(self.epochs, self.generator_losses)
        self.gen_val_line.set_data(self.epochs, self.val_generator_losses)
        self.disc_line.set_data(self.epochs, self.discriminator_losses)
        self.disc_val_line.set_data(self.epochs, self.val_discriminator_losses)
        self.ax[0].relim()
        self.ax[0].autoscale_view()

        # Subplot 1: Generator components
        if self.gx_losses or self.val_gx_losses:
            self.gx_line.set_data(self.epochs[:len(self.gx_losses)], self.gx_losses)
            self.gx_val_line.set_data(self.epochs[:len(self.val_gx_losses)], self.val_gx_losses)
        if self.gy_losses or self.val_gy_losses:
            self.gy_line.set_data(self.epochs[:len(self.gy_losses)], self.gy_losses)
            self.gy_val_line.set_data(self.epochs[:len(self.val_gy_losses)], self.val_gy_losses)
        self.ax[1].relim()
        self.ax[1].autoscale_view()

        # Subplot 2: Discriminator components
        if self.dx_losses or self.val_dx_losses:
            self.dx_line.set_data(self.epochs[:len(self.dx_losses)], self.dx_losses)
            self.dx_val_line.set_data(self.epochs[:len(self.val_dx_losses)], self.val_dx_losses)
        if self.dy_losses or self.val_dy_losses:
            self.dy_line.set_data(self.epochs[:len(self.dy_losses)], self.dy_losses)
            self.dy_val_line.set_data(self.epochs[:len(self.val_dy_losses)], self.val_dy_losses)
        self.ax[2].relim()
        self.ax[2].autoscale_view()

        # Subplot 3: Cycle Consistency Losses
        if self.cycle_losses:
            self.cycle_loss_line.set_data(self.epochs, self.cycle_losses)
            self.cycle_loss_val_line.set_data(self.epochs, self.val_cycle_losses)
        self.ax[3].relim()
        self.ax[3].autoscale_view()

        # Subplot 4: Identity Losses
        if self.identity_losses:
            self.identity_loss_line.set_data(self.epochs, self.identity_losses)
            self.identity_loss_val_line.set_data(self.epochs, self.val_identity_losses)
        self.ax[4].relim()
        self.ax[4].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
