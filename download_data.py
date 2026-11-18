import os
import sys
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TARGET_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
ZIP_FILE = os.path.join(DATA_DIR, "sms_spam_collection.zip")

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_TSV_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TARGET_FILE):
        print(f"[INFO] Dataset already exists at: {TARGET_FILE}")
        return True

    print("[INFO] Attempting to download from UCI Machine Learning Repository...")
    try:
        # User-Agent header helps avoid 403 blocks on some academic servers
        req = urllib.request.Request(
            UCI_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(ZIP_FILE, "wb") as out_f:
            out_f.write(resp.read())

        print("[INFO] Downloaded zip. Extracting SMSSpamCollection...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extract("SMSSpamCollection", DATA_DIR)

        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)

        print(f"[SUCCESS] Dataset saved to: {TARGET_FILE}")
        return True
    except Exception as e:
        print(f"[WARNING] Failed downloading from UCI: {e}")

    print("[INFO] Attempting to download from backup mirror...")
    try:
        req = urllib.request.Request(
            MIRROR_TSV_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(TARGET_FILE, "wb") as out_f:
            out_f.write(resp.read())

        print(f"[SUCCESS] Dataset saved from mirror to: {TARGET_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed downloading from mirror: {e}")

    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit UCI Repository: https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
    print("   Or Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print("2. Download and place the file named 'SMSSpamCollection' in the 'data/' folder.")
    print("=" * 60 + "\n")
    return False

if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
