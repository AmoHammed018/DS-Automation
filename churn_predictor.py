
import pandas as pd
import numpy as np
import joblib
import os


class ChurnPredictor:
    """
    Load a trained churn model pipeline and generate churn predictions.
    """

    def __init__(self, model_path='churn_model.joblib'):
        """
        Initialize the predictor by loading the saved model pipeline.
        """
        self.pipeline = joblib.load(model_path)
        self.training_probs = None

    def set_training_probs(self, training_probs):
        """
        Store training probability predictions for percentile calculation.
        """
        self.training_probs = np.asarray(training_probs)

    @staticmethod
    def prepare_data(df):
        """
        Prepare Week 2-style churn data for prediction.
        """

        df = df.copy()

        # Convert TotalCharges to numeric if necessary
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'],
                errors='coerce'
            )

        # Remove identifier
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])

        # Create engineered numerical features
        if 'AvgMonthlyCharge' not in df.columns:
            df['AvgMonthlyCharge'] = (
                df['TotalCharges'] /
                df['tenure'].replace(0, np.nan)
            )

        if 'TenureMonthlyRatio' not in df.columns:
            df['TenureMonthlyRatio'] = (
                df['tenure'] /
                df['MonthlyCharges'].replace(0, np.nan)
            )

        if 'LifetimeValue' not in df.columns:
            df['LifetimeValue'] = (
                df['MonthlyCharges'] *
                df['tenure']
            )

        # Create categorical tenure bins
        if 'TenureBin' not in df.columns:
            df['TenureBin'] = pd.cut(
                df['tenure'],
                bins=[-1, 6, 12, 24, 48, 72],
                labels=[
                    '0-6',
                    '6-12',
                    '12-24',
                    '24-48',
                    '48-72'
                ]
            )

        # Create categorical charge bins
        if 'ChargeBin' not in df.columns:
            df['ChargeBin'] = pd.cut(
                df['MonthlyCharges'],
                bins=[0, 30, 50, 70, 90, 120],
                labels=[
                    '0-30',
                    '30-50',
                    '50-70',
                    '70-90',
                    '90-120'
                ]
            )

        # Remove columns that were not used by the trained model
        if 'charge_per_tenure' in df.columns:
            df = df.drop(columns=['charge_per_tenure'])

        # Remove target if supplied
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])

        # Required model feature order
        expected_cols = [
            'tenure',
            'PhoneService',
            'Contract',
            'PaymentMethod',
            'MonthlyCharges',
            'TotalCharges',
            'AvgMonthlyCharge',
            'TenureMonthlyRatio',
            'LifetimeValue',
            'TenureBin',
            'ChargeBin'
        ]

        # Check that all required columns exist
        missing_cols = [
            col for col in expected_cols
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required model features: {missing_cols}"
            )

        return df[expected_cols]

    def predict(self, df, prepare=True):
        """
        Generate churn probabilities and predictions.
        """

        if prepare:
            model_data = self.prepare_data(df)
        else:
            model_data = df.copy()

        # Generate churn probabilities
        churn_prob = self.pipeline.predict_proba(
            model_data
        )[:, 1]

        results = df.copy()

        results['Churn_Probability'] = churn_prob

        results['Churn_Prediction'] = (
            results['Churn_Probability'] >= 0.5
        ).astype(int)

        # Calculate percentile relative to training probabilities
        if self.training_probs is not None:
            results['Churn_Percentile'] = (
                results['Churn_Probability']
                .apply(
                    lambda x:
                    (self.training_probs < x).mean() * 100
                )
            )

        return results

    def predict_file(self, file_path):
        """
        Load a CSV file and generate churn predictions.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.

        Returns
        -------
        pandas.DataFrame
            Prediction results.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        data = pd.read_csv(file_path)

        print(f"Loaded file: {file_path}")
        print(f"Rows: {len(data)}")
        print(f"Columns: {len(data.columns)}")

        return self.predict(data, prepare=True)

    def print_predictions(self, df, prepare=True):
        """
        Print prediction results for a DataFrame.
        """

        results = self.predict(
            df,
            prepare=prepare
        )

        print("\nChurn Predictions:")
        print(
            results[
                [
                    'Churn_Probability',
                    'Churn_Prediction'
                ]
            ]
        )

        if 'Churn_Percentile' in results.columns:
            print(
                "\nChurn Percentiles "
                "(relative to training distribution):"
            )

            print(
                results[
                    [
                        'Churn_Probability',
                        'Churn_Percentile'
                    ]
                ]
            )

        return results

    def predict_file_interactive(self):
        """
        Ask the user for a CSV file path and generate predictions.
        """

        file_path = input(
            "Enter the path to the churn CSV file "
            "[default: new_churn_data.csv]: "
        ).strip()

        if not file_path:
            file_path = 'new_churn_data.csv'

        results = self.predict_file(file_path)

        self.print_predictions_from_results(results)

        return results

    @staticmethod
    def print_predictions_from_results(results):
        """
        Print an existing prediction DataFrame.
        """

        print("\nPrediction Results:")
        print(
            results[
                [
                    'Churn_Probability',
                    'Churn_Prediction'
                ]
            ]
        )

        if 'Churn_Percentile' in results.columns:
            print(
                "\nChurn Percentiles:"
            )

            print(
                results[
                    [
                        'Churn_Probability',
                        'Churn_Percentile'
                    ]
                ]
            )


if __name__ == '__main__':

    predictor = ChurnPredictor(
        'churn_model.joblib'
    )

    # Allow the user to specify the input CSV file
    file_path = input(
        "Enter the path to the churn CSV file "
        "[default: new_churn_data.csv]: "
    ).strip()

    if not file_path:
        file_path = 'new_churn_data.csv'

    results = predictor.predict_file(
        file_path
    )

    predictor.print_predictions_from_results(
        results
    )
