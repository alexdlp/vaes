import mlflow
from mlflow.tracking import MlflowClient
import mlflow.tensorflow
from tensorflow.keras.callbacks import Callback
from itp_fabad.utils.model_helpers import get_model_summary
from itp_fabad.logger import logger

from typing import List, Dict

# Desactiva cualquier autolog implícito
mlflow.tensorflow.autolog(disable=True)
mlflow.autolog(disable=True)   

      
def get_model_summary(model: tf.keras.Model) -> str:
    """
    Generates a textual summary of the model.

    Args:
        model (tf.keras.Model): The model for which the summary will be generated.

    Returns:
        str: The model summary as a formatted string.
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)


class MLFlowModelCheckpoint(Callback):
    def __init__(self, monitor='val_loss', save_best_only=True, mode='min', input_signature=None,patience: int = 0, *args, **kwargs):
        """
        Custom callback to log the best model in MLflow.
        
        Args:
            monitor (str): Metric to monitor for improvement (e.g., 'val_loss').
            save_best_only (bool): If True, only the best model will be logged to MLflow.
            mode (str): 'min' to minimize the monitored metric, 'max' to maximize it.
            input_signature (tf.Tensor or None): Input example for model signature. If None, signature will not be set.
            *args, **kwargs: Additional arguments for the callback.
        """
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.input_signature = input_signature  
        self.min_delta = 1e-4  # Minimum change in the monitored metric to qualify as an improvement
        self.patience = patience
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.client: MlflowClient = MlflowClient()
        self.run_id: str | None = None
        super(MLFlowModelCheckpoint, self).__init__(*args, **kwargs)

    def on_train_begin(self, logs: Dict | None = None) -> None:
        """Cache the current run_id at training start."""
        run = mlflow.active_run()
        if run is None:
            raise RuntimeError("No active MLflow run found.")
        self.run_id = run.info.run_id

    def _purge_logged_models(self) -> List[str]:
        """
        Delete every Logged-Model (``models/m-<hash>``) that belongs to the
        current run.  Returns the list of deleted ``model_id`` values.
        """
        deleted: List[str] = []
        if self.run_id is None:
            return deleted
        

        registered_models = mlflow.search_logged_models(
            filter_string=f"source_run_id='{self.run_id}'",
            output_format="list")

        for lm in registered_models:
            try:
                model_id = lm if isinstance(lm, str) else getattr(lm, "model_id", None)
                if model_id is None:
                    logger.warning(f"Skipping entry with missing model_id: {lm}")
                    continue

                self.client.delete_logged_model(model_id)
                deleted.append(model_id)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Could not delete model '{lm}': {exc}")

        return deleted
 

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs the model in MLflow only if the model has improved.
        """
        if logs is None:
            logger.warning("logs is None at epoch end")
            return
        
        # Wait for `patience` epochs before starting to log
        if epoch + 1 < self.patience:
            return

        if not self.save_best_only:
            # Si no es save_best_only, no hacemos nada o logueamos siempre
            # (tu código original no logueaba aquí, lo cual es correcto si no se quiere)
            return

        current = logs.get(self.monitor)
        if current is None:
            logger.error(f"❌ Metric '{self.monitor}' not found in logs. Available keys: {list(logs.keys())}")
            return

        # Define improvement condition
        improved = (self.mode == 'min' and current < self.best - self.min_delta) or \
                (self.mode == 'max' and current > self.best + self.min_delta)

        if improved:

            # Purge previous logged models belonging to this run
            removed = self._purge_logged_models()
            if removed:
                logger.info(f"Removed logged models from run: {removed}")
            
            mlflow.tensorflow.log_model(
                model=self.model,
                artifact_path="best_model",  
                registered_model_name=None, 
                step = epoch,
                input_example=self.input_signature
            )

            logger.info(
                f"✅ Epoch {epoch + 1}: {self.monitor} improved "
                f"from {self.best:.6f} to {current:.6f}. New best model logged."
                )
            
            self.best = current


class MLFlowModelSummaryCallback(Callback):
    def __init__(self, config, *args, **kwargs):
        """
        Callback to log hyperparameters, model summary, and layer activations.

        Args:
            config (dict or DictConfig): The configuration object containing all hyperparameters.
        """
        super().__init__(*args, **kwargs)
        self.cfg = config 

    def parse_hyperparameters(self):
        """
        Parse the configuration object and extract hyperparameters into a dictionary.
        
        Returns:
            dict: A dictionary containing the hyperparameters to log.
        """
        hyperparameters = {}

        # 1. Model parameters
        model_params = self.cfg.get('model', {})
        for key, value in model_params.items():
            hyperparameters[f"model_{key}"] = value

        # 2. Optimizer parameters
        optimizer_params = self.cfg.get('optimizer', {})
        for key, value in optimizer_params.items():
            hyperparameters[f"optimizer_{key}"] = value

        # 3. Data parameters
        data_params = self.cfg.get('data', {})
        hyperparameters["train_fraction"] = data_params.get('train', {}).get('fraction', None)
        hyperparameters["val_fraction"] = data_params.get('val', {}).get('fraction', None)
        hyperparameters["max_seq_len"] = data_params.get('max_seq_len', None)
        hyperparameters["feature_dim"] = data_params.get('feature_dim', None)

        # 4. Training parameters
        training_params = self.cfg.get('training', {})
        for key, value in training_params.items():
            hyperparameters[f"training_{key}"] = value

        # 5. Callbacks parameters
        callbacks_params = self.cfg.get('callbacks', {})
        early_stopping = callbacks_params.get('early_stopping', {})
        model_checkpoint = callbacks_params.get('model_checkpoint', {})

        for key, value in early_stopping.items():
            hyperparameters[f"early_stopping_{key}"] = value
        
        for key, value in model_checkpoint.items():
            hyperparameters[f"model_checkpoint_{key}"] = value

        return hyperparameters
    
    
    def log_model_hparams(self):

        # Parse the hyperparameters from the configuration and log them to mlflow
        hyperparameters = self.parse_hyperparameters()  
        mlflow.log_params(hyperparameters)  

        if hyperparameters.get("model_name") == "autoencoder":
        
            # Access the encoder and decoder models directly
            encoder = self.model.get_layer("model_encoder")  
            decoder = self.model.get_layer("model_decoder")  

            # Log the encoder and decoder summaries as text artifacts in MLflow
            mlflow.log_text(get_model_summary(encoder), "encoder_summary.txt")  
            mlflow.log_text(get_model_summary(decoder), "decoder_summary.txt")

            # Log the activations of the layers as parameters
            # mlflow.log_param("encoder_layer_activations", str(get_layer_activations(encoder)))
            # mlflow.log_param("decoder_layer_activations", str(get_layer_activations(decoder)))
        
        else:
            model_name = hyperparameters.get("model_name") 
            # Log the model summary as a text artifact in MLflow
            mlflow.log_text(get_model_summary(self.model), f"{model_name}_summary.txt")  

    def on_epoch_begin(self, epoch, logs=None):
        """
        Logs the model summary, activations of the layers, and hyperparameters at the beginning of training.
        """
        if epoch == 0:
            logger.info("Logging model hyperparameters before training...")
            self.log_model_hparams()
             
    def on_epoch_end(self, epoch, logs=None):
        """
        Log metrics at the end of each epoch.
        """
        if logs:
            for metric, value in logs.items():
                mlflow.log_metric(metric, value, step=epoch)  # Log each metric (e.g., loss, accuracy)

        optimizer = self.model.optimizer
        
        if optimizer is None:
            logger.warning(f"Epoch {epoch}: No optimizer found.")
            return

        # Extraer correctamente el LR dependiendo de su estructura
        try:
            lr = optimizer.learning_rate.numpy()
        except AttributeError:
            lr = float(optimizer.learning_rate)
        except Exception as e:
            logger.error(f"Error accessing learning rate: {e}")
            lr = None

        if lr is not None:
            mlflow.log_metric("learning_rate", lr, step=epoch)

