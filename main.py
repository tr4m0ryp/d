import pandas as pd
import numpy as np

def load_dataset(file_path):
    """
    Load the dataset from a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset from {file_path}.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def recognize_pattern_and_improve(df):
    """
    Recognize patterns in the dataset and improve it by casting data types,
    handling missing values, and ensuring consistency.
    """
    # Define expected patterns
    expected_data_types = {
        'input_columns': np.float32,  # Replace 'input_columns' with your actual column names
        'label_column': np.int32,     # Replace 'label_column' with your actual label column name
    }

    # Print initial data types
    print("\nInitial Data Types:")
    print(df.dtypes)

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values Per Column:")
    print(missing_values)

    # Fill missing values with the median (or another strategy)
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if expected_data_types.get(column) == np.int32:
                df[column].fillna(df[column].median(), inplace=True)
                print(f"Filled missing values in column '{column}' with median value.")
            elif expected_data_types.get(column) == np.float32:
                df[column].fillna(df[column].median(), inplace=True)
                print(f"Filled missing values in column '{column}' with median value.")

    # Print data types and convert to expected types
    print("\nConverting data types to expected types:")
    for column in df.columns:
        if df[column].dtype != np.int32 and df[column].dtype != np.float32:
            try:
                df[column] = df[column].astype(np.float32)  # Try converting to float32 first
                print(f"Converted column '{column}' to float32.")
            except ValueError as e:
                print(f"Error converting column '{column}' to float32: {e}")
                print("Attempting to coerce invalid values...")
                df[column] = pd.to_numeric(df[column], errors='coerce')
                df[column].fillna(df[column].median(), inplace=True)
                df[column] = df[column].astype(np.float32)
                print(f"Coerced and converted column '{column}' to float32.")

    # Detect and handle outliers (optional)
    for column in df.columns:
        if df[column].dtype in [np.int32, np.float32]:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
            print(f"Handled outliers in column '{column}' using IQR method.")

    return df

def save_improved_dataset(df, output_file_path):
    """
    Save the improved dataset to a CSV file.
    """
    try:
        df.to_csv(output_file_path, index=False)
        print(f"Improved dataset saved to {output_file_path}.")
    except Exception as e:
        print(f"Error saving improved dataset: {str(e)}")

if __name__ == "__main__":
    # Specify the file path to your dataset
    input_file_path = 'C:\\Users\\Moussa\\Downloads\\Pump-and-Dump-Detection-on-Cryptocurrency-master\\Pump-and-Dump-Detection-on-Cryptocurrency-master\\TargetCoinPrediction\\FeatGeneration\\feature\\test_sample.csv'
    output_file_path = 'improved_test_sample.csv'  # Output file path for the improved dataset

    # Load the dataset
    df = load_dataset(input_file_path)

    # If the dataset was successfully loaded, improve it
    if df is not None:
        improved_df = recognize_pattern_and_improve(df)
        
        # Save the improved dataset to a new file
        save_improved_dataset(improved_df, output_file_path)
