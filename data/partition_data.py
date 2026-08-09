import pandas as pd
import os

def partition_dataco_data():
    file_path = 'data/DataCoSupplyChainDataset.csv'
    
    print("Loading DataCo dataset...")
    # Using 'latin1' encoding as the DataCo dataset often contains special characters
    df = pd.read_csv(file_path, encoding='latin1')
    
    print(f"Total dataset shape: {df.shape}")
    
    # Client 1: Europe Market
    client_1_df = df[df['Market'] == 'Europe']
    client_1_path = 'data/client_1_data.csv'
    client_1_df.to_csv(client_1_path, index=False)
    print(f"Client 1 (Europe) data saved: {client_1_df.shape} rows.")
    
    # Client 2: LATAM Market
    client_2_df = df[df['Market'] == 'LATAM']
    client_2_path = 'data/client_2_data.csv'
    client_2_df.to_csv(client_2_path, index=False)
    print(f"Client 2 (LATAM) data saved: {client_2_df.shape} rows.")

if __name__ == "__main__":
    partition_dataco_data()