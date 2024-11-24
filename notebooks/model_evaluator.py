import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from matplotlib import cm
from itertools import cycle
import numpy as np
import seaborn as sns

class ModelEvaluator:
    def __init__(self, task_type='binary_classification'):
        """
        Initializes the ModelEvaluator with an empty DataFrame to store results
        and a list to store model outputs for plotting.

        Parameters:
        - task_type (str): Type of task ('binary_classification', 'multi_class_classification', 'regression').
                           Default is 'binary_classification'.
        """
        # Validate Evaluation task_type
        supported_tasks = ['binary_classification', 'multi_class_classification', 'regression']
        if task_type not in supported_tasks:
            raise ValueError(f"Unsupported task_type. Choose from {supported_tasks}.")

        self.task_type = task_type  # Store task_type for later use

        # Initialize an empty DataFrame to store results
        if self.task_type in ['binary_classification', 'multi_class_classification']:
            self.results_df = pd.DataFrame(columns=[
                'Model', 'Dataset', 'Accuracy', 'Precision', 'Recall',
                'F1-Score', 'ROC AUC', 'Average Precision'
            ])
        elif self.task_type == 'regression':
            self.results_df = pd.DataFrame(columns=[
                'Model', 'Dataset', 'Mean Squared Error', 'R2 Score'
            ])

        # Initialize a list to store outputs for plotting later
        self.model_outputs = []
        # self.FIGURE_SIZE = (7,5)

        

    # ------------------------------ Evaluation ------------------------------
    def evaluate_model(self, model_name, y_actual, y_pred, y_pred_prob=None, y_pred_dec=None, dataset_name='Test'):
        """
        Evaluates the model based on the actual and predicted labels or values,
        appends or updates results in the internal DataFrame, and returns the metrics as a one-row DataFrame.

        Parameters:
        - model_name (str): Name of the model.
        - y_actual (array-like): True labels or values for the dataset.
        - y_pred (array-like): Predicted labels or values for the dataset.
        - y_pred_prob (array-like, optional): Predicted probabilities for classification tasks.
        - dataset_name (str): Name of the dataset (e.g., 'Train', 'Validation', 'Test').

        Returns:
        - pd.DataFrame: A one-row DataFrame containing all evaluation metrics for the model.
                        The model name is set as the index.
        """
        # If y_pred_prob is None, initialise it with y_pred_dec instead
        if y_pred_prob is None:
            y_pred_prob = y_pred_dec
            
        # Initialize metrics dictionary with None
        metrics = {col: None for col in self.results_df.columns}
        metrics.update({'Model': model_name, 'Dataset': dataset_name})

        # ------------------------------ Actual Metrics ------------------------------
        if self.task_type in ['binary_classification', 'multi_class_classification']:
            # Classification-specific metrics
            average = 'binary' if self.task_type == 'binary_classification' else 'macro'
            metrics['Accuracy'] = accuracy_score(y_actual, y_pred)
            metrics['Precision'] = precision_score(y_actual, y_pred, average=average, zero_division=0)
            metrics['Recall'] = recall_score(y_actual, y_pred, average=average, zero_division=0)
            metrics['F1-Score'] = f1_score(y_actual, y_pred, average=average, zero_division=0)

            # Optional AUC metrics for classification with probabilities
            if y_pred_prob is not None:
                if self.task_type == 'binary_classification':
                    metrics['ROC AUC'] = roc_auc_score(y_actual, y_pred_prob)
                    metrics['Average Precision'] = average_precision_score(y_actual, y_pred_prob)
                else:
                    try:
                        metrics['ROC AUC'] = roc_auc_score(y_actual, y_pred_prob, multi_class='ovr', average='macro')
                        metrics['Average Precision'] = average_precision_score(y_actual, y_pred_prob, average='macro')
                    except ValueError:
                        pass

        elif self.task_type == 'regression':
            # Regression-specific metrics
            from sklearn.metrics import mean_squared_error, r2_score
            metrics['Mean Squared Error'] = mean_squared_error(y_actual, y_pred)
            metrics['R2 Score'] = r2_score(y_actual, y_pred)

        # ------------------------------ Handle duplicate evaluation ------------------------------
        # Find existing row to update if present
        existing_row_index = self.results_df[
            (self.results_df['Model'] == model_name) &
            (self.results_df['Dataset'] == dataset_name)
        ].index

        if not existing_row_index.empty:
            # Update the existing row directly by setting values per column
            for key in metrics:
                self.results_df.at[existing_row_index[0], key] = metrics[key]
        else:
            # Append a new row if no existing entry is found
            self.results_df = pd.concat([self.results_df, pd.DataFrame([metrics])], ignore_index=True)

        # Search for existing duplicate output
        existing_output = next((i for i, output in enumerate(self.model_outputs)
                                if output['model_name'] == model_name and output['dataset_name'] == dataset_name), None)

        new_output = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'y_actual': y_actual,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob,
            'task_type': self.task_type
        }

        if existing_output is not None:
            # Update the existing output if there is
            self.model_outputs[existing_output] = new_output
        else:
            # Append a new output
            self.model_outputs.append(new_output)

        # Prepare the return DataFrame with relevant metrics
        return pd.DataFrame([{k: metrics[k] for k in metrics if k not in ['Model', 'Dataset']}], index=[model_name])

    # ------------------------------ Full Results ------------------------------
    def display_results(self, dataset_name=None, model_name=None):
        """
        Retrieves the results DataFrame, optionally filtered by dataset and/or model.

        Parameters:
        - dataset_name (str, optional): Name of the dataset to filter by.
        - model_name (str, optional): Name of the model to filter by.

        Returns:
        - pd.DataFrame: Filtered DataFrame containing evaluation metrics, 
                        with irrelevant metrics removed based on task_type.
        """
        df = self.results_df.copy()

        # Apply filters for dataset and model name
        if dataset_name:
            df = df[df['Dataset'] == dataset_name]
        if model_name:
            df = df[df['Model'] == model_name]

        # If dataset_name is specified, drop the 'Dataset' column to avoid redundancy
        if dataset_name and 'Dataset' in df.columns:
            df = df.drop(columns=['Dataset'])

        # Remove any columns with all None values to clean up display
        df = df.dropna(axis=1, how='all')

        # Reset index for clean display
        return df.reset_index(drop=True)

    # ------------------------------ Curve Plot ------------------------------
    def plot_curves(self, curve_type='roc', dataset_name=None, model_names=None):
        """
        Plots ROC or Precision-Recall curves based on the specified type.
        Allows filtering by dataset and/or model.

        Parameters:
        - curve_type (str): Type of curve to plot ('roc' or 'precision_recall').
        - dataset_name (str, optional): Name of the dataset to filter by.
        - model_names (list of str, optional): List of model names to include.

        Returns:
        - None: Displays the plot.
        """
        if curve_type not in ['roc', 'precision_recall']:
            raise ValueError("curve_type must be either 'roc' or 'precision_recall'")

        # plt.figure(figsize=self.FIGURE_SIZE)
        plt.figure()
        colors = cm.tab10.colors  # Use tab10 colormap for distinct colors
        color_idx = 0  # Initialize color index

        # # Baseline lines
        # if curve_type == 'roc':
        #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        # elif curve_type == 'precision_recall':
        #     plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--', label='Baseline')

        # Iterate over model outputs and plot accordingly
        for output in self.model_outputs:
            model_name = output['model_name']
            current_dataset = output['dataset_name']
            task_type = output.get('task_type', 'binary_classification')

            # Apply filters
            if dataset_name and current_dataset != dataset_name:
                continue
            if model_names and model_name not in model_names:
                continue

            y_actual = output['y_actual']
            y_pred_prob = output.get('y_pred_prob')

            if y_pred_prob is None:
                continue  # Skip if no probability predictions

            color = colors[color_idx % len(colors)]
            color_idx += 1  # Increment color index

            if task_type == 'binary_classification':
                if curve_type == 'roc':
                    fpr, tpr, _ = roc_curve(y_actual, y_pred_prob)
                    roc_auc = roc_auc_score(y_actual, y_pred_prob)
                    plt.plot(fpr, tpr, lw=2, label=f"{model_name} ({current_dataset}) AUC = {roc_auc:.2f}", color=color)
                elif curve_type == 'precision_recall':
                    precision, recall, _ = precision_recall_curve(y_actual, y_pred_prob)
                    avg_precision = average_precision_score(y_actual, y_pred_prob)
                    plt.plot(recall, precision, lw=2, label=f"{model_name} ({current_dataset}) AP = {avg_precision:.2f}", color=color)

            elif task_type == 'multi_class_classification':
                # Assuming y_pred_prob is of shape (n_samples, n_classes)
                classes = np.unique(y_actual)
                if len(classes) < 3:
                    # Treat as binary if less than 3 classes
                    if curve_type == 'roc':
                        fpr, tpr, _ = roc_curve(y_actual, y_pred_prob[:,1])
                        roc_auc = roc_auc_score(y_actual, y_pred_prob[:,1])
                        plt.plot(fpr, tpr, lw=2, label=f"{model_name} ({current_dataset}) AUC = {roc_auc:.2f}", color=color)
                    elif curve_type == 'precision_recall':
                        precision, recall, _ = precision_recall_curve(y_actual, y_pred_prob[:,1])
                        avg_precision = average_precision_score(y_actual, y_pred_prob[:,1])
                        plt.plot(recall, precision, lw=2, label=f"{model_name} ({current_dataset}) AP = {avg_precision:.2f}", color=color)
                    continue

                # Binarize the output
                y_bin = label_binarize(y_actual, classes=classes)
                n_classes = y_bin.shape[1]
                colors_cycle = cycle(cm.tab10.colors)
                for i in range(n_classes):
                    color_class = next(colors_cycle)
                    if curve_type == 'roc':
                        try:
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                            roc_auc = roc_auc_score(y_bin[:, i], y_pred_prob[:, i])
                            plt.plot(fpr, tpr, lw=1, label=f"{model_name} ({current_dataset}) Class {classes[i]} AUC = {roc_auc:.2f}", color=color_class)
                        except ValueError:
                            continue  # Skip if ROC AUC cannot be computed for the class
                    elif curve_type == 'precision_recall':
                        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_prob[:, i])
                        avg_precision = average_precision_score(y_bin[:, i], y_pred_prob[:, i])
                        plt.plot(recall, precision, lw=1, label=f"{model_name} ({current_dataset}) Class {classes[i]} AP = {avg_precision:.2f}", color=color_class)

        # Final plot adjustments
        if curve_type == 'roc':
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            title = 'ROC Curves' + (f' for {dataset_name} Set' if dataset_name else ' for All Datasets')
        elif curve_type == 'precision_recall':
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            title = 'Precision-Recall Curves' + (f' for {dataset_name} Set' if dataset_name else ' for All Datasets')

        plt.title(title, fontsize=16)
        # plt.legend(loc='lower right', fontsize=10)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid(True)
        plt.show()

    # ------------------------------ Confusion Matrix Plot ------------------------------
    def plot_confusion_matrix(self, model_name, dataset_name='Test', normalize=False):
        """
        Plots the confusion matrix for a specified model and dataset with enhanced styling.

        Parameters:
        - model_name (str): Name of the model.
        - dataset_name (str): Name of the dataset (default is 'Test').
        - normalize (bool): If True, displays percentages instead of raw counts.

        Returns:
        - None: Displays the confusion matrix plot.
        """
        # Find the corresponding model output
        relevant_outputs = [
            output for output in self.model_outputs
            if output['model_name'] == model_name and output['dataset_name'] == dataset_name
        ]

        if not relevant_outputs:
            print(f"No data found for model '{model_name}' on dataset '{dataset_name}'.")
            return

        output = relevant_outputs[-1]  # Use the latest entry if multiple exist

        # Ensure task_type is classification
        if self.task_type not in ['binary_classification', 'multi_class_classification']:
            print("Confusion matrix is only applicable for classification tasks.")
            return

        y_actual = output['y_actual']
        y_pred = output['y_pred']

        # Compute confusion matrix
        cm_matrix = confusion_matrix(y_actual, y_pred)
        classes = np.unique(y_actual)

        # Normalize the confusion matrix if requested
        if normalize:
            cm_matrix = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        # plt.figure(figsize=self.FIGURE_SIZE)
        plt.figure()

        sns.heatmap(cm_matrix, annot=True, fmt=".2f" if normalize else "d",
                    cmap="Blues", cbar=True, xticklabels=classes, yticklabels=classes,
                    linewidths=0.5, linecolor='gray')

        plt.title(f'Confusion Matrix for {model_name} ({dataset_name} Set)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)

        # Adjust font size for axis labels and title
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)

        # Ensure layout is tight so labels fit well
        plt.tight_layout()
        plt.show()

