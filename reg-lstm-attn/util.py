import numpy as np
from keras import callbacks
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from IPython.display import display, update_display


def metrics(model, xtest, true, y_per_engine, mean_metrics=True):
    """
    Prints regression metrics: MSE, RMSE, MAE, R2

    Returns:
        prediction for each engine: either mean or final
    """
    prediction = model.predict(xtest).reshape(-1)

    i, engine_predictions = 0, []
    for num_y in y_per_engine:
        engine_predictions.append(prediction[i:i+num_y])
        i += num_y

    # performance using average of the predictions of last 5 test records per engine
    mean_prediction = [np.average(engine) for engine in engine_predictions]

    if mean_metrics:
        MSE = mean_squared_error(true, mean_prediction)
        print('MSE: ', MSE)
        RMSE = np.sqrt(MSE)
        print('RMSE: ', RMSE)
        MAE = mean_absolute_error(true, mean_prediction)
        print('MAE: ', MAE)
        r2 = r2_score(true, mean_prediction)
        print('R2: ', r2)
        return mean_prediction
    
    
    # performance only using the last test record per engine
    last_predictions = [predictions[0] for predictions in engine_predictions]

    MSE = mean_squared_error(true, last_predictions)
    print('MSE (Taking only last sample): ', MSE)
    RMSE = np.sqrt(MSE)
    print('RMSE (Taking only last sample): ', RMSE)
    MAE = mean_absolute_error(true, last_predictions)
    print('MAE (Taking only last sample): ', MAE)
    r2 = r2_score(true, last_predictions)
    print('R2 (Taking only last sample): ', r2)
    return last_predictions
    

def test_plot(model_name, predictions, true, mean_predictions=True):
    """
    Graphs predictions against true labels. Default will title the graph as mean predictions.

    Returns:
        fig, ax of the plot
    """
    fig = plt.figure(figsize=(16, 6), dpi=200)
    ax = plt.axes()
    ax.plot(predictions, marker='o', color='cornflowerblue', label='Predicted RUL')
    ax.plot(true, marker='o', color='orangered', label='True RUL')
    ax.set_title(f'{model_name}: Testing True RUL vs {"Mean" if mean_predictions else "Final"} Predicted RUL')
    ax.set_xlabel('Machine Number')
    ax.set_ylabel('RUL: Cycles Left')
    ax.grid(linestyle=':')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig, ax


class LossCurve(callbacks.Callback):
    """
    Dynamically graphs and updates training and validation loss curves during model training.
    
    To be imported and used in a Jupyter Notebook ONLY.
    """
    def __init__(self, fig: Figure, axes: list[Axes]):
        super().__init__()
        self.fig = fig
        self.axes = axes
        self.display_id = None
        self.train_loss = []
        self.val_loss = []
    
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        self.x = np.arange(1, self.total_epochs+1)
        self.fig.suptitle(f'Epoch: 0/{self.total_epochs} | Training and Validation Loss Curves (MSE)')
        self.train_line, = self.axes[0].plot([], self.train_loss, marker='o', color='deepskyblue', label='Train Loss')
        self.val_line, = self.axes[0].plot([], self.val_loss, marker='o', color='crimson', label='Validation Loss')
        self.val_line2, = self.axes[1].plot([], self.val_loss, marker='o', color='crimson', label='Validation Loss')
        for ax in self.axes:
            ax.set_xlim(0, self.total_epochs+1)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss (MSE)')
            ax.legend(loc='upper right')
            ax.grid(linestyle=':')
        if self.display_id is None:
            self.display_id = display(self.fig, display_id=True).display_id
        self.fig.tight_layout()
            
    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        
        self.train_line.remove()
        self.val_line.remove()
        self.val_line2.remove()
        
        self.train_line, = self.axes[0].plot(self.x[:epoch+1], self.train_loss, marker='o', color='deepskyblue', label='Train Loss')
        self.val_line, = self.axes[0].plot(self.x[:epoch+1], self.val_loss, marker='o', color='crimson', label='Validation Loss')
        self.val_line2, = self.axes[1].plot(self.x[:epoch+1], self.val_loss, marker='o', color='crimson', label='Validation Loss')
        
        self.fig.suptitle(f'Epoch: {epoch+1}/{self.total_epochs} | Training and Validation Loss Curves (MSE)')
        update_display(self.fig, display_id=self.display_id)
        
    def on_train_end(self, logs=None):
        plt.close(self.fig)


class RULGraph(callbacks.Callback):
    """
    Visualizes model performance on a portion of input data after every epoch.
    
    To be imported and used in a Jupyter Notebook ONLY.
    """
    def __init__(self, fig: Figure, ax: Axes, x: np.ndarray, y: np.ndarray, time_points :int=3000):
        super().__init__()
        self.fig = fig
        self.ax = ax
        self.x = x[:time_points]
        self.y = y[:time_points]
        self.time_points = time_points
        self.display_id = None
        self.prediction_lines = []
            
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        self.cmap = ListedColormap(cm.winter(np.linspace(0, 1, self.total_epochs)))
        norm = plt.Normalize(1, self.total_epochs)
        sm = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        self.fig.colorbar(mappable=sm, ax=self.ax, ticks=np.arange(0, self.total_epochs+1, 3), label='Epochs')   
        self.true_line, = self.ax.plot(self.y, label='True RUL', color='orangered')
        self.ax.legend(loc='upper right')
        self.fig.tight_layout()
        self.ax.set_title(f'Epoch: 0/{self.total_epochs} | Predicted RUL vs True RUL for {self.time_points} Training Sequences')
        self.ax.set_xlabel('Sequence')
        self.ax.set_ylabel('RUL: Cycles Left')
        if self.display_id is None:
            self.display_id = display(self.fig, display_id=True).display_id
            
    def on_epoch_end(self, epoch, logs=None):
        if len(self.prediction_lines) > 0:
            self.prediction_lines[-1].set_zorder(self.prediction_lines[-1].get_zorder()-1)
            self.prediction_lines[-1].set_alpha(0.2)
        
        prediction = self.model.predict(self.x).reshape(-1)
        prediction_line, = self.ax.plot(prediction, label=f'Predicted RUL, Epoch {epoch+1}', color=self.cmap(epoch), zorder=epoch)
        self.prediction_lines.append(prediction_line)
        
        self.true_line.set_zorder(epoch-1)
        
        self.ax.legend(handles=[self.true_line, prediction_line], loc='upper right').set_zorder(self.total_epochs+1)
        self.ax.set_title(f'Epoch: {epoch+1}/{self.total_epochs} | Predicted RUL vs True RUL for {self.time_points} Training Sequences')
        update_display(self.fig, display_id=self.display_id)
        
    def on_train_end(self, logs=None):
        plt.close(self.fig)
