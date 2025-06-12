import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import warnings

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    def preprocess_banking_data(df):
        """
        Melakukan preprocessing otomatis pada banking dataset:
        1. Menghapus outlier pada kolom 'previous'
        2. Mapping kolom 'yes'/'no' menjadi 1/0 pada kolom tertentu
        3. One-hot encoding pada kolom kategorikal (kecuali target)
        4. Standardisasi fitur numerik (kecuali target)
        5. Mengembalikan dataframe siap latih
        """
        # 1. Remove outliers for 'previous'
        Q1 = df['previous'].quantile(0.01)
        Q3 = df['previous'].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['previous'] >= lower_bound) & (df['previous'] <= upper_bound)].copy()

        # 2. Map 'yes'/'no' columns to 1/0
        yes_no_cols = ['default', 'housing', 'loan', 'y']
        for col in yes_no_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0})

        # 3. One-hot encoding for categorical columns except target
        categorical_cols = df.select_dtypes(include='object').columns
        categorical_cols_no_target = [col for col in categorical_cols if col != 'y']
        df = pd.get_dummies(df, columns=categorical_cols_no_target, drop_first=True)

        # 4. Standardize numeric columns except target
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('y')
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df

    with mlflow.start_run():
        # Load dataset
        df = pd.read_csv('dataset/banking-data.csv', sep=';')

        # Preprocess the data
        df_processed = preprocess_banking_data(df)

        # Log the processed DataFrame as an artifact
        output_path = 'preprocessing/banking-data_preprocessing.csv'
        df_processed.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)

        print(f"Preprocessing completed and data saved as '{output_path}'.")

        mlflow.log_param("preprocessing_steps", "outlier_removal, yes_no_mapping, one_hot_encoding, standardization")
        mlflow.log_metric("num_rows", df_processed.shape[0])
        mlflow.log_metric("num_columns", df_processed.shape[1])

        print("Data preprocessing and logging completed successfully.")
