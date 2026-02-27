import mlflow
import numpy as np
import re

import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from itp_fabad.utils.model_helpers import unpad_series
from itp_fabad.logger.logger import logger
from pathlib import Path
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

from itp_fabad.data_processing.processing import load_and_process_cube_csv, layer_segmentation_and_feature_extraction


class EvaluationCallback(Callback):
    """
    Custom Keras callback to evaluate the model at the end of each epoch.
    Loads validation data from CSVs and statistics from multiple .npz files.
    """

    def __init__(self, validation_base_path: str, uncertainty_data_dir:str, standarization_data_dir: str):
        """
        Initializes the callback.

        Args:
            validation_base_path (str): Base directory containing validation data.
        """
        super().__init__()
        self.validation_base_path = Path(validation_base_path)
        self.uncertainty_data_dir = Path(uncertainty_data_dir)
        self.standarization_data_dir = Path(standarization_data_dir)

        self.validation_data = {
            'ref1': {'Cube_2': {}, 'Cube_4': {}, 'Cube_9': {}}, 
            'ref2': {'Cube_2': {}, 'Cube_4': {}, 'Cube_9': {}}
        }

        # Generate 3 random layer indices (same for both references)
        self.selected_layer_indices = self._generate_random_layer_indices()

        # Generate paths for the selected layers
        self._generate_csv_paths()
        self._load_and_select_vectors()
        self._load_uncertainty_data()
        self._load_standardization_data()

    def _get_mlflow_path(self, filename: str) -> str:
        """
        Gets the local path for saving artifacts in MLflow.
        """
        artifact_uri = Path(mlflow.active_run().info.artifact_uri)
        artifact_uri_local = artifact_uri.as_posix().replace('file:', '')  # Remove 'file:' if needed
        artifact_path = Path(artifact_uri_local) / filename
        return artifact_path.as_posix()
    
    def _load_uncertainty_data(self):
        """Loads data from the .npz file and assigns it to the corresponding attributes."""
        if not self.uncertainty_data_dir.exists():
            raise FileNotFoundError(f"The file {self.uncertainty_data_dir} does not exist.")
        
        data = np.load(self.uncertainty_data_dir)

        # Assigning values to class attributes
        try:
            self.global_means = data["median"]
            self.upper_limits = data["upper_bound"]
            self.lower_limits = data["lower_bound"]

        except KeyError as e:
            raise ValueError(f"The .npz file does not contain the expected key: {e}")

        # Dimension validation
        if not (self.global_means.shape == self.upper_limits.shape == self.lower_limits.shape):
            raise ValueError("The dimensions of means, upper_limits, and lower_limits do not match.")
        
    def _load_standardization_data(self) -> None:
        """
        Loads standardization data from a file.
        """
        logger.info(f"Loading standardization stats from {self.standarization_data_dir.name}")
        stats = np.load(self.standarization_data_dir, allow_pickle=True)
        self.sequences_mean = stats["sequences_mean"]
        self.sequences_std = stats["sequences_std"]
        # self.features_mean = stats["features_mean"]
        # self.features_std = stats["features_std"]
        # self.feature_names = list(stats["feature_names"])

    def _generate_random_layer_indices(self):
        """
        Generates 3 random layer indices for each cube.
        """
        selected_indices = {}

        for cube in [2, 4, 9]:
            ## EXCLUDE 0 FROM RANDOM CHOICE SINCE THERE IS NO LAYER 0
            selected_indices[cube] = np.random.choice(np.arange(1, 309), 3, replace=False).tolist()

        return selected_indices
    
    def _generate_csv_paths(self):
        """
        Generates full paths to the selected CSV files for each cube and reference.
        Stores them inside `self.validation_data`.
        """
        ## 
        for ref in ["ref1", "ref2"]:
            for cube in [2, 4, 9]:
                for layer in self.selected_layer_indices[cube]:
                    self.validation_data[ref][f'Cube_{cube}'][f'Layer_{layer}'] = {
                        'path': self.validation_base_path / ref / "vl" / f"Cube_{cube}" / f"Cube_{cube}_Layer_{layer}.csv"
                    }

                    #print(self.validation_data[ref][f'Cube_{cube}'][f'Layer_{layer}']['path'])

    def _get_cube_t0(self, csv_path: Path) -> float:
        """
        Obtains the start time (t0) of a given cube by extracting the cube number from the provided path
        and loading the data from the first layer (Layer_1) CSV file of that cube.

        To do this, the method:
        - Extracts the cube number using regex from the filename, expecting a path format like:
        ".../Cube_{cube_number}_Layer_{layer_number}.csv".
        - Modifies the original path by replacing "Layer_{layer_number}" with "Layer_1" to point directly
        to the first layer CSV.
        - Loads the first-layer CSV data and retrieves the printing start time ("Start time / us") of the 
        first printed point.

        Args:
            csv_path (Path): Path to the CSV file with format:
                            ".../Cube_{cube_number}_Layer_{layer_number}.csv".

        Returns:
            float: The cube start time (cube_t0), corresponding to the start time of the first printed point.
        """
        # Obtener el número del cubo con regex:
        cube_match = re.search(r'Cube_(\d+)', csv_path.name)
        if not cube_match:
            raise ValueError(f"No cube number found in path: {csv_path}")

        cube_number = int(cube_match.group(1))

        # Reemplazar Layer_X por Layer_1 directamente
        layer_1_path = Path(re.sub(r'Layer_\d+', 'Layer_1', str(csv_path)))

        # Cargar CSV y obtener cube_t0
        layer_df = load_and_process_cube_csv(layer_1_path, cube_number)
        cube_t0 = layer_df["Start time / us"].iloc[0]

        return cube_t0

    def _load_and_select_vectors(self):
        """
        Loads layer data, selects random vectors, and stores the processed results in `self.validation_data`.
        The selected indices are the same for both references (`ref1` and `ref2`).
        """

        for cube in [2, 4, 9]:
            for layer in self.selected_layer_indices[cube]: 

                paths = {
                    "ref1": self.validation_data["ref1"][f'Cube_{cube}'][f"Layer_{layer}"]["path"],
                    "ref2": self.validation_data["ref2"][f'Cube_{cube}'][f"Layer_{layer}"]["path"]
                }
  
                # Check if both CSV files exist
                if not paths["ref1"].exists() or not paths["ref2"].exists():
                    logger.warning(f"⚠️ Warning: CSV file not found for Cube {cube} - Layer {layer}")
                    continue

                selected_vectors = {}

                # 🔹 Step 1: Load the CSV layer data for both references
                layer_df_ref1 = load_and_process_cube_csv(paths["ref1"], cube)
                layer_df_ref2 = load_and_process_cube_csv(paths["ref2"], cube)

                # 🔹 Step 2: Extract feature vectors from both references
                full_vectors_ref1 = layer_segmentation_and_feature_extraction(info_layer=layer_df_ref1, layer_idx = layer, cube_idx = cube, cube_t0 = self._get_cube_t0(paths["ref1"]))
                full_vectors_ref2 = layer_segmentation_and_feature_extraction(info_layer=layer_df_ref2, layer_idx = layer, cube_idx = cube, cube_t0 = self._get_cube_t0(paths["ref2"]))

                num_vectors = len(full_vectors_ref1)

                if num_vectors < 3:
                    logger.warning(f"⚠️ Warning: Layer {layer} of Cube {cube} has only {num_vectors} vectors.")
                    selected_indices = np.arange(num_vectors)  # Take all available vectors
                else:
                    selected_indices = np.random.choice(num_vectors, 3, replace=False)

                selected_vectors = {
                    "ref1": {f"Vector_{idx}": full_vectors_ref1[idx] for idx in selected_indices},
                    "ref2": {f"Vector_{idx}": full_vectors_ref2[idx] for idx in selected_indices}
                }

                # 🔹 Store results in validation_data
                self.validation_data["ref1"][f'Cube_{cube}'][f"Layer_{layer}"].update(selected_vectors["ref1"])
                self.validation_data["ref2"][f'Cube_{cube}'][f"Layer_{layer}"].update(selected_vectors["ref2"])

                #print(f"✅ Cube {cube} - Layer {layer}: {selected_indices} vectors selected.")

    def preprocess_series(self, series: np.ndarray, target_length: int) -> np.ndarray:
        """
        Interpolates and standardizes the input series.
        """
        original_length = len(series)

        # Interpolate to the target length
        x_original = np.linspace(0, 1, original_length)
        x_interp = np.linspace(0, 1, target_length)


        # Return secuence if already correct size
        if original_length == target_length:
            return series

        # Ensure sequence has at least 2 points
        if original_length < 2:
            logger.warning("Sequence has less than 2 points; returning a constant sequence.")
            return np.full(target_length, series[0])

        interp_func = interp1d(x_original, series, kind="linear", fill_value="extrapolate")

        interpolated_series = interp_func(x_interp)

        # Standardize
        processed_series = (interpolated_series - self.sequences_mean) / self.sequences_std
       

        return processed_series
    
    def postprocess_prediction(self, predicted_series: np.ndarray, original_length: int) -> np.ndarray:
        """
        De-standardizes and de-interpolates the predicted series.
        """

        # DESCALE BEFORE DE-INTERPOLATION

        # De-standardize
        descaled_series = (predicted_series * self.sequences_std) + self.sequences_mean

        # De-interpolate back to original length
        x_pred = np.linspace(0, 1, len(descaled_series))
        x_original = np.linspace(0, 1, original_length)
        interp_func = interp1d(x_pred, descaled_series, kind="linear", fill_value="extrapolate")

        postprocessed_series = interp_func(x_original)

        return postprocessed_series
    
    def predict_series(self, series: np.ndarray) -> np.ndarray:
        """
        Processes a series by interpolating, standardizing, predicting, and then restoring to original scale.
        """

        preprocessed_series = self.preprocess_series(series, target_length=600)
        #print('preporcessed series', preprocessed_series.shape)
        
        # Ensure the shape expected by the model (batch_size=1, timesteps=600, features=1)
        input_series = preprocessed_series.reshape(1, -1, 1)
        #print('input series', input_series.shape)

        # Predict
        predicted_series = self.model.predict(input_series, verbose=0).squeeze() # Ajuste según la salida del modelo
        #print('predicted series', predicted_series.shape)
        
        # Post-process prediction (de-standardization and de-interpolation)
        reconstructed_series = self.postprocess_prediction(predicted_series, original_length=len(series))

        return reconstructed_series
    
    def _generate_plots(self, epoch: int) -> str:
        """
        Generates a PDF with validation plots comparing ref1 and ref2.
        """
        pdf_filename = f"validation_plots_epoch_{epoch}.pdf"
        pdf_path = self._get_mlflow_path(pdf_filename)

        with PdfPages(pdf_path) as pdf:
            for cube in [2, 4, 9]:
                for layer in self.selected_layer_indices[cube]:
                    layer_key = f"Layer_{layer}"

                    # Verificar que existen datos en ambas referencias
                    if layer_key not in self.validation_data["ref1"][f'Cube_{cube}'] or \
                    layer_key not in self.validation_data["ref2"][f'Cube_{cube}']:
                        continue

                    for vector_key in self.validation_data["ref1"][f'Cube_{cube}'][layer_key].keys():
                        if vector_key.startswith("Vector_"):

                            ref1_series = np.array(self.validation_data["ref1"][f'Cube_{cube}'][layer_key][vector_key]['meltpool_seq'])
                            ref2_series = np.array(self.validation_data["ref2"][f'Cube_{cube}'][layer_key][vector_key]['meltpool_seq'])
                      
                            if len(ref1_series) == 0 or len(ref2_series) == 0:
                                continue
                            
                            samples = len(ref1_series)
                            steps = np.arange(samples)

                            baseline = (ref1_series + ref2_series) / 2
                            upper_limit = baseline + unpad_series(self.upper_limits[layer-1])[0:samples] - unpad_series(self.global_means[layer-1])[0:samples]
                            lower_limit = baseline - unpad_series(self.global_means[layer-1])[0:samples] + unpad_series(self.lower_limits[layer-1])[0:samples]
                            fill_between_x = steps[0:samples]

                            predicted_ref1 = self.predict_series(ref1_series)
                            predicted_ref2 = self.predict_series(ref2_series)
                            predicted_mean = self.predict_series((ref1_series+ref2_series)/2)

                            #logger.info(f"Layer {layer}: Original ref samples {len(ref1_series)} | baseline samples {len(baseline)} ")

                            y_min_upper = int(min(ref1_series.min(), ref2_series.min(), predicted_ref1.min(), predicted_ref2.min()) * 1.2)
                            y_max_upper = int(max(ref1_series.max(), ref2_series.max(), predicted_ref1.max(), predicted_ref2.max()) * 1.1)  
                            y_min_upper = -50 if y_min_upper == 0 else y_min_upper  
                                             

                            # Crear la figura con dos subplots
                            fig, axs = plt.subplots(2, 3, figsize=(20, 12), sharey=False, sharex = False)
                            fig.suptitle(f"Cube {cube} | Layer {layer} | {vector_key} - Interpolated Series",fontsize=16, y = 0.95)

                            axs[0,0].plot(ref1_series, linestyle="-", color="blue", alpha=0.7, label = "Original Ref1")
                            axs[0,0].plot(predicted_ref1, linestyle="-", color="#66B2FF", alpha=1, label="Autoencoder Predicted Ref1")
                            axs[0,0].plot(baseline, linestyle="dotted", color="black", alpha=0.3)
                            axs[0,0].fill_between(x = fill_between_x,
                                                    y1 = upper_limit, 
                                                    y2 = lower_limit, 
                                                    color='gray', alpha=0.15, label='Uncertainty (95% CI)')

                            axs[0,0].set_title("Reference 1", fontsize=14)
                            axs[0,0].set_xlabel("Time Step", fontsize=12)
                            axs[0,0].set_ylabel("Meltpool Value", fontsize=12)
                            axs[0,0].grid(alpha=0.3)
                            axs[0,0].legend()

                            axs[0,1].plot(ref2_series, linestyle="-", color="red", alpha=0.7, label = "Original Ref2")
                            axs[0,1].plot(predicted_ref2, linestyle="-", color="#CC4C4C", alpha=1, label="Autoencoder Predicted Ref2")
                            axs[0,1].plot(baseline, linestyle="dotted", color="black", alpha=0.3)
                            axs[0,1].fill_between(x = fill_between_x,
                                                y1 = upper_limit, 
                                                y2 = lower_limit, 
                                                color='gray', alpha=0.15, label='Uncertainty (95% CI)')
                            
                            axs[0,1].set_title("Reference 2", fontsize=14)
                            axs[0,1].set_xlabel("Time Step", fontsize=12)
                            axs[0,1].set_ylabel("Meltpool Value", fontsize=12)
                            axs[0,1].grid(alpha=0.3)
                            axs[0,1].legend()

                            axs[0,2].plot((ref1_series+ref2_series)/2, linestyle="-", color="#2E8B57", alpha=0.7, label="Original Ref1&Ref2 Mean")
                            axs[0,2].plot(predicted_mean, linestyle="-", color="#66CDAA", alpha=1, label="Autoencoder Predicted Mean")
                            axs[0,2].plot(baseline, linestyle="dotted", color="black", alpha=0.3)
                            axs[0,2].fill_between(x = fill_between_x,
                                                    y1 = upper_limit, 
                                                    y2 = lower_limit, 
                                                    color='gray', alpha=0.15, label='Uncertainty (95% CI)')

                            axs[0,2].set_title("Mean", fontsize=14)
                            axs[0,2].set_xlabel("Time Step", fontsize=12)
                            axs[0,2].set_ylabel("Meltpool Value", fontsize=12)
                            axs[0,2].grid(alpha=0.3)
                            axs[0,2].legend()

                            # Assign limits to the upper plots (original vs. predicted)
                            axs[0, 0].set_ylim(y_min_upper, y_max_upper)
                            axs[0, 1].set_ylim(y_min_upper, y_max_upper)
                            axs[0, 2].set_ylim(y_min_upper, y_max_upper)

                            # 🔹 Second row: Difference and Residuals
                            predicted_diff = predicted_ref1 - predicted_ref2
                            residuals_ref1 = ref1_series - predicted_ref1
                            residuals_ref2 = ref2_series - predicted_ref2

                            mean_residuals = ((ref1_series+ref2_series)/2) - predicted_mean
     
                            # Compute axis limits for lower plots (difference & residuals)
                            y_min_lower = int(min(predicted_diff.min(), residuals_ref1.min(), residuals_ref2.min()) * 1.1)
                            y_max_lower = int(max(predicted_diff.max(), residuals_ref1.max(), residuals_ref2.max()) * 1.1)

                            axs[1, 0].plot(predicted_diff, linestyle="-", color="purple", alpha=0.8, label="Difference (Pred1 - Pred2)")
                            axs[1, 0].axhline(0, color="black", linestyle="--", alpha=0.5)
                            axs[1, 0].set_title("Difference between Predicted Series", fontsize=14)
                            axs[1, 0].set_xlabel("Time Step", fontsize=12)
                            axs[1, 0].set_ylabel("Difference Value", fontsize=12)
                            axs[1, 0].grid(alpha=0.3)
                            axs[1, 0].legend()

                            axs[1, 1].plot(residuals_ref1, linestyle="-", color="blue", alpha=0.7, label="Residuals Ref1 (Orig1 - Pred1)")
                            axs[1, 1].plot(residuals_ref2, linestyle="-", color="red", alpha=0.7, label="Residuals Ref2 (Orig2 - Pred2)")
                            axs[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5)
                            axs[1, 1].set_title("Residuals (Original - Predicted)", fontsize=14)
                            axs[1, 1].set_xlabel("Time Step", fontsize=12)
                            axs[1, 1].set_ylabel("Residual Value", fontsize=12)
                            axs[1, 1].grid(alpha=0.3)
                            axs[1, 1].legend()

                            axs[1, 2].plot(mean_residuals, linestyle="-", color="#2E8B57", alpha=0.7, label="Residuals Mean (Orig - Predicted)")
                            axs[1, 2].axhline(0, color="black", linestyle="--", alpha=0.5)
                            axs[1, 2].set_title("Residuals (Orig Mean - Predicted Mean)", fontsize=14)
                            axs[1, 2].set_xlabel("Time Step", fontsize=12)
                            axs[1, 2].set_ylabel("Residual Value", fontsize=12)
                            axs[1, 2].grid(alpha=0.3)
                            axs[1, 2].legend()

                            # Assign limits to the lower plots (difference & residuals)
                            axs[1, 0].set_ylim(y_min_lower, y_max_lower)
                            axs[1, 1].set_ylim(y_min_lower, y_max_lower)

                            # Guardar en PDF
                            pdf.savefig(fig)
                            plt.close(fig)

        logger.info(f"📄 PDF saved: {pdf_path}")
        return pdf_path
    
    def on_train_begin(self, logs=None):
        """
        Ensure the model is available when training starts.
        """
        if self.model is None:
            raise ValueError("Model is not set. Ensure this callback is used inside model.fit()")
        
        logger.info("✅ Model assigned to EvaluationCallback.")

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Called at the end of each epoch to generate validation plots.
        """

        if epoch == 0 or epoch % 10 == 0:
            logger.info(f"📊 Generating validation plots for epoch {epoch}...")
            pdf_path = self._generate_plots(epoch)
            logger.info(f"📁 Validation plots saved at: {pdf_path}")


    def on_train_end(self, logs=None):
        """
        Callback function executed when training ends.
        """
        total_epochs = self.params.get("epochs", None)
        logger.info(f"🏁 Training completed after {total_epochs} epochs. Generating final validation plots...")

        pdf_path = self._generate_plots(epoch=total_epochs)
        logger.info(f"📁 Final validation plots saved at: {pdf_path}")

    