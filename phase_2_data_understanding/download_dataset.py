"""
phase_2_data_understanding/download_dataset.py
-----------------------------------------------
Downloads **only the real Tomato images** from the PlantVillage dataset
on Kaggle (abdallahalidev/plantvillage-dataset) and copies them into a
local destination directory.

The Kaggle zip unpacks as:
    plantvillage dataset/
        color/           <- RGB originals   <- we use this by default
            Tomato___Bacterial_spot/
            Tomato___Early_blight/
            ...
        segmented/
        grayscale/

CLI usage
---------
    # Requires a valid ~/.kaggle/kaggle.json token:
    python -m phase_2_data_understanding.download_dataset

    # Custom destination or image variant:
    python -m phase_2_data_understanding.download_dataset \\
        --dest data/PlantVillage --image_type color

Kaggle credentials
------------------
Create a token at https://www.kaggle.com/settings -> API -> "Create New Token".
Place the downloaded kaggle.json at  ~/.kaggle/kaggle.json  (chmod 600).

In Google Colab:
    from google.colab import files
    files.upload()               # select kaggle.json
    !mkdir -p ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
"""

import os
import shutil
import argparse


# ── Exact Tomato class folder names in abdallahalidev/plantvillage-dataset ────
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]

# Kaggle dataset slug for PlantVillage
DEFAULT_KAGGLE_SLUG = "abdallahalidev/plantvillage-dataset"


# ── Real download ─────────────────────────────────────────────────────────────
def download_real(
    dest: str = "data/PlantVillage",
    kaggle_slug: str = DEFAULT_KAGGLE_SLUG,
    image_type: str = "color",
    crop: str = "Tomato",
) -> None:
    """
    Downloads the PlantVillage dataset from Kaggle, locates the
    ``image_type`` subfolder (color | segmented | grayscale), then copies
    only the folders matching ``crop`` into ``dest``.

    Parameters
    ----------
    dest        : Target directory (created if needed).
    kaggle_slug : Kaggle dataset identifier (owner/dataset-name).
    image_type  : Which image variant to use ('color', 'segmented', or
                  'grayscale').  Defaults to 'color'.
    crop        : Crop name prefix to filter (default 'Tomato').
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "kaggle package not found.  Install it with:  pip install kaggle"
        )

    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    # Stage to a temp dir so dest stays clean on failure
    raw_dir = os.path.join("data", "_plantvillage_raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(
        f"Downloading '{kaggle_slug}' -> {raw_dir}  "
        "(this may take several minutes -- the full zip is ~4 GB)..."
    )
    api.dataset_download_files(kaggle_slug, path=raw_dir, unzip=True)
    print("Download complete.\n")

    # ── Locate the chosen image-type subfolder ────────────────────────────────
    # Expected layout after extraction:
    #   raw_dir/
    #     plantvillage dataset/          <- top-level dir name (has a space)
    #       color/
    #         Tomato___Bacterial_spot/   <- class folders
    #         ...
    #       segmented/
    #       grayscale/
    image_root = _find_image_type_root(raw_dir, image_type, crop)
    if image_root is None:
        raise FileNotFoundError(
            f"Could not find a '{image_type}' subfolder containing '{crop}' "
            f"class folders inside {raw_dir}.\n"
            "Check that the Kaggle slug is correct and extraction succeeded."
        )

    print(f"Using image folder: {image_root}")
    _copy_crop_folders(image_root, dest, crop)
    print(f"\n[OK] {crop} leaf folders copied to: {dest}")
    _print_summary(dest)


def _find_image_type_root(base: str, image_type: str, crop: str) -> str | None:
    """
    Walk the extracted tree and return the path of the first directory whose
    name matches ``image_type`` (case-insensitive) and that contains at
    least one class folder whose name starts with ``crop``.

    Falls back to any directory directly containing crop-prefixed subfolders
    if the named subfolder is not found.
    """
    image_type_lower = image_type.lower()
    crop_lower = crop.lower()
    fallback = None

    for dirpath, dirnames, _ in os.walk(base):
        has_crop = any(d.lower().startswith(crop_lower) for d in dirnames)
        if os.path.basename(dirpath).lower() == image_type_lower and has_crop:
            return dirpath
        if has_crop and fallback is None:
            fallback = dirpath

    return fallback


def _copy_crop_folders(src_root: str, dest: str, crop: str = "Tomato") -> None:
    """
    Copy only the class folders matching ``crop`` from ``src_root`` into
    ``dest``.  Skips folders that already exist (allows safe resume).
    """
    crop_lower = crop.lower()
    os.makedirs(dest, exist_ok=True)
    copied = 0
    skipped = 0

    for folder in sorted(os.listdir(src_root)):
        if not folder.lower().startswith(crop_lower):
            continue
        src_path = os.path.join(src_root, folder)
        dest_path = os.path.join(dest, folder)

        if not os.path.isdir(src_path):
            continue

        if os.path.exists(dest_path):
            n = sum(
                1 for f in os.listdir(dest_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            print(f"  Skipping (already exists, {n} images): {folder}")
            skipped += 1
            continue

        n_imgs = len([
            f for f in os.listdir(src_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"  Copying: {folder}  ({n_imgs} images)")
        shutil.copytree(src_path, dest_path)
        copied += 1

    print(f"\nCopied {copied} class folder(s), skipped {skipped} already-present.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _print_summary(dest: str) -> None:
    """Print a class-by-class image count table."""
    print(f"\n{'Class':<60} {'Images':>6}")
    print("-" * 68)
    total = 0
    for cls in sorted(os.listdir(dest)):
        cls_dir = os.path.join(dest, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = sum(
            1 for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"  {cls:<58} {n:>6}")
        total += n
    print("-" * 68)
    print(f"  {'TOTAL':<58} {total:>6}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download the real PlantVillage Tomato dataset from Kaggle.\n"
            "Requires a valid ~/.kaggle/kaggle.json API token.\n\n"
            "Get your token at: "
            "https://www.kaggle.com/settings -> API -> Create New Token"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest", default="data/PlantVillage",
        help="Destination directory for the class folders. "
             "(default: data/PlantVillage)"
    )
    parser.add_argument(
        "--kaggle_slug", default=DEFAULT_KAGGLE_SLUG,
        help=f"Kaggle dataset slug (owner/dataset-name). "
             f"(default: {DEFAULT_KAGGLE_SLUG})"
    )
    parser.add_argument(
        "--image_type", default="color",
        choices=["color", "segmented", "grayscale"],
        help="Image variant to use from the zip. (default: color)"
    )
    parser.add_argument(
        "--crop", default="Tomato",
        help="Crop prefix to filter class folders. (default: Tomato)"
    )
    args = parser.parse_args()

    download_real(
        dest=args.dest,
        kaggle_slug=args.kaggle_slug,
        image_type=args.image_type,
        crop=args.crop,
    )
