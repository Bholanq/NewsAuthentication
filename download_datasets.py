import os
import zipfile
import kaggle

# Define variables
dataset = "clmentbisaillon/fake-and-real-news-dataset"
dataset_zip = "fake-and-real-news-dataset.zip"
target_dir = ".venv"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Download dataset using Kaggle API
def download_kaggle_dataset():
    print("⬇Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files(dataset, path=target_dir, unzip=False)
    print("Download complete.")

# Extract the ZIP file
def extract_zip():
    zip_path = os.path.join(target_dir, dataset_zip)
    if not os.path.exists(zip_path):
        print(f"Dataset ZIP file not found at {zip_path}.")
        return

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")

# Main function
if __name__ == "__main__":
    download_kaggle_dataset()
    extract_zip()
