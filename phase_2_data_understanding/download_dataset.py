"""
phase_2_data_understanding/download_dataset.py
-----------------------------------------------
Supports two download modes:

  1. REAL  — Downloads the full PlantVillage dataset from Kaggle
             (abdallahalidev/plantvillage-dataset) and filters it to
             only keep a chosen crop's folders (default: Tomato).

  2. MOCK  — Generates tiny synthetic leaf images for quick pipeline
             testing without any credentials or internet access.

CLI usage
---------
    # Real dataset, tomato leaves only (requires Kaggle credentials):
    python -m phase_2_data_understanding.download_dataset \
        --mode real --crop Tomato --dest data/PlantVillage

    # Mock dataset (default, no credentials needed):
    python -m phase_2_data_understanding.download_dataset --mode mock

Kaggle credentials
------------------
Create a token at https://www.kaggle.com/settings → API → "Create New Token".
Place the downloaded kaggle.json at  ~/.kaggle/kaggle.json  (chmod 600).
In Colab, upload it via the file browser or run:

    from google.colab import files
    files.upload()               # select kaggle.json
    !mkdir -p ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
"""

import os
import shutil
import argparse
from PIL import Image, ImageDraw


# ── Tomato class names present in abdallahalidev/plantvillage-dataset ────────
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Miner",
    "Tomato___Mosaic_virus",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]


# ── Real download ─────────────────────────────────────────────────────────────
def download_real(
    dest: str = "data/PlantVillage",
    crop: str = "Tomato",
    kaggle_slug: str = "abdallahalidev/plantvillage-dataset",
) -> None:
    """
    Downloads PlantVillage from Kaggle and copies only the folders whose
    names start with ``crop`` (case-insensitive) into ``dest``.

    Parameters
    ----------
    dest        : Target directory (will be created if needed).
    crop        : Crop prefix to keep, e.g. "Tomato", "Potato", "Corn".
    kaggle_slug : Kaggle dataset identifier (owner/dataset-name).
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "kaggle package not found.  Install it with:  pip install kaggle"
        )

    print(f"Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    # Download to a temp staging dir so we don't pollute dest on failure
    raw_dir = os.path.join("data", "_plantvillage_raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading '{kaggle_slug}' → {raw_dir}  (this may take a few minutes)...")
    api.dataset_download_files(kaggle_slug, path=raw_dir, unzip=True)
    print("Download complete.")

    # The zip unpacks as:  raw_dir/plantvillage dataset (color)/  (or similar)
    # Walk to find the deepest folder that contains class subdirs
    extracted_root = _find_image_root(raw_dir, crop)
    if extracted_root is None:
        raise FileNotFoundError(
            f"Could not find any '{crop}' folders inside {raw_dir}.  "
            f"Check that the Kaggle slug '{kaggle_slug}' is correct."
        )

    print(f"Found dataset root: {extracted_root}")
    _copy_crop_folders(extracted_root, dest, crop)
    print(f"\n✅  Done!  Tomato leaf folders copied to: {dest}")
    _print_summary(dest)


def _find_image_root(base: str, crop: str) -> str | None:
    """
    Walk the extracted directory tree and return the first directory that
    contains at least one sub-folder whose name starts with ``crop``.
    """
    crop_lower = crop.lower()
    for dirpath, dirnames, _ in os.walk(base):
        if any(d.lower().startswith(crop_lower) for d in dirnames):
            return dirpath
    return None


def _copy_crop_folders(src_root: str, dest: str, crop: str) -> None:
    """Copy only folders matching ``crop`` from src_root into dest."""
    crop_lower = crop.lower()
    os.makedirs(dest, exist_ok=True)
    copied = 0
    for folder in sorted(os.listdir(src_root)):
        if not folder.lower().startswith(crop_lower):
            continue
        src_path  = os.path.join(src_root, folder)
        dest_path = os.path.join(dest, folder)
        if not os.path.isdir(src_path):
            continue
        if os.path.exists(dest_path):
            print(f"  Skipping (already exists): {folder}")
            continue
        print(f"  Copying: {folder}  ({len(os.listdir(src_path))} images)")
        shutil.copytree(src_path, dest_path)
        copied += 1
    print(f"\nCopied {copied} class folder(s) to {dest}")


# ── Mock dataset ──────────────────────────────────────────────────────────────
def setup_mock_dataset(
    base_dir: str = "data/PlantVillage",
    num_samples_per_class: int = 12,
) -> None:
    """
    Creates tiny synthetic JPEG leaf images for pipeline smoke-testing.
    No internet or Kaggle credentials required.
    """
    classes = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"]
    os.makedirs(base_dir, exist_ok=True)

    print(f"Setting up mock dataset in {base_dir}...")
    for class_name in classes:
        class_path = os.path.join(base_dir, class_name)
        os.makedirs(class_path, exist_ok=True)

        for i in range(num_samples_per_class):
            img_path = os.path.join(class_path, f"leaf_{i}.jpg")
            if os.path.exists(img_path):
                continue

            img  = Image.new("RGB", (256, 256), color=(101, 67, 33))
            draw = ImageDraw.Draw(img)
            draw.ellipse([50, 50, 206, 206], fill=(34, 139, 34))

            if class_name == "Tomato___Early_blight":
                draw.ellipse([80,  80,  100, 100], fill=(139, 69, 19))
                draw.ellipse([130, 120, 150, 140], fill=(139, 69, 19))
            elif class_name == "Tomato___Late_blight":
                draw.ellipse([70,  70,  110, 110], fill=(80, 50, 20))
                draw.ellipse([110, 140, 160, 190], fill=(80, 50, 20))

            img.save(img_path, "JPEG")

    print("Mock dataset setup complete.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _print_summary(dest: str) -> None:
    """Print a class-by-class image count table."""
    print(f"\n{'Class':<55} {'Images':>6}")
    print("-" * 63)
    total = 0
    for cls in sorted(os.listdir(dest)):
        cls_dir = os.path.join(dest, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = sum(
            1 for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"  {cls:<53} {n:>6}")
        total += n
    print("-" * 63)
    print(f"  {'TOTAL':<53} {total:>6}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download or mock the PlantVillage dataset."
    )
    parser.add_argument(
        "--mode", choices=["real", "mock"], default="mock",
        help="'real' requires Kaggle credentials; 'mock' is always available."
    )
    parser.add_argument(
        "--crop", default="Tomato",
        help="Crop to filter when mode=real (e.g. Tomato, Potato, Corn)."
    )
    parser.add_argument(
        "--dest", default="data/PlantVillage",
        help="Destination directory for the dataset."
    )
    parser.add_argument(
        "--kaggle_slug", default="abdallahalidev/plantvillage-dataset",
        help="Kaggle dataset slug (owner/dataset-name)."
    )
    parser.add_argument(
        "--mock_samples", type=int, default=12,
        help="Images per class when mode=mock."
    )
    args = parser.parse_args()

    if args.mode == "real":
        download_real(dest=args.dest, crop=args.crop, kaggle_slug=args.kaggle_slug)
    else:
        setup_mock_dataset(base_dir=args.dest, num_samples_per_class=args.mock_samples)
