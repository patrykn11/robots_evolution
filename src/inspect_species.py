import zipfile
import pickle
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))

def inspect_species_zip(zip_path: str) -> None:
    print(f"Inspecting {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = sorted(zip_ref.namelist())
            print(f"Found {len(file_list)} files in zip.")
            print(f"First 5 files: {file_list[:5]}")
            
            pkl_files = [f for f in file_list if f.endswith('.pkl') and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('._')]
            if not pkl_files:
                print("No valid .pkl files found.")
                return

            first_pkl = pkl_files[0]
            print(f"Loading {first_pkl}...")
            with zip_ref.open(first_pkl) as file:
                data = pickle.load(file)
                print(f"Type: {type(data)}")
                if hasattr(data, '__dict__'):
                    print(f"Attributes: {data.__dict__.keys()}")
                elif isinstance(data, dict):
                    print(f"Keys: {data.keys()}")
                else:
                    print(f"Data: {data}")
                
                if hasattr(data, 'body'):
                    print(f"Body shape: {data.body.shape}")
                    print(f"Body content (unique values): {set(data.body.flatten())}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_species_zip("results/species_experiment/species_zip/species.zip")
