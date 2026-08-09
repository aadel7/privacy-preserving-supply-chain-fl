import pandas as pd
import os

def partition_data():
    print("Loading raw DataCo dataset...")
    # Adjust path if your raw data file has a specific name
    raw_path = "data/DataCoSupplyChainDataset.csv"  # update if named differently in your data/ folder
    if not os.path.exists(raw_path):
        # Look for any csv file in data/
        csv_files = [f for f in os.listdir("data") if f.endswith(".csv") and f != "data.csv"]
        if csv_files:
            raw_path = os.path.join("data", csv_files[0])
        else:
            raise FileNotFoundError("Could not find the raw DataCo dataset CSV in the data/ folder.")

    df = pd.read_csv(raw_path, encoding='latin1')
    print(f"Loaded dataset with {len(df)} rows.")

    # Check unique shipping modes
    if 'Shipping Mode' in df.columns:
        print("Shipping modes found:", df['Shipping Mode'].unique())
    else:
        raise KeyError("Column 'Shipping Mode' not found in dataset.")

    # Create strongly non-IID partitions based on Shipping Mode
    express_modes = ['First Class', 'Same Day']
    standard_modes = ['Second Class', 'Standard Class']

    df_client1 = df[df['Shipping Mode'].isin(express_modes)].copy()
    df_client2 = df[df['Shipping Mode'].isin(standard_modes)].copy()

    print(f"Client 1 (Express Hub - {express_modes}): {len(df_client1)} rows")
    print(f"Client 2 (Standard Hub - {standard_modes}): {len(df_client2)} rows")

    # Ensure output directories exist or save directly where docker expects them
    # Assuming your docker volumes mount data from a specific path or client folders
    os.makedirs("data/client_1", exist_ok=True)
    os.makedirs("data/client_2", exist_ok=True)

    df_client1.to_csv("data/client_1/data.csv", index=False)
    df_client2.to_csv("data/client_2/data.csv", index=False)
    
    print("Partitioning complete! Non-IID silos saved to data/client_1/ and data/client_2/")

if __name__ == "__main__":
    partition_data()