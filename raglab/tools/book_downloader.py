import requests
from pathlib import Path

def download_file(url: str, save_path: str):
    response = requests.get(url)
    response.raise_for_status()  # Raise error for bad status
    Path(save_path).write_text(response.text, encoding="utf-8")
    print(f"✅ Downloaded and saved to {save_path}")

if __name__ == "__main__":
    url = "https://www.gutenberg.org/cache/epub/2680/pg2680.txt"
    save_path = "data/meditations_marcus_aurelius.txt"
    download_file(url, save_path)
