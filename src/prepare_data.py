import zipfile
import os
import shutil
import pandas as pd

base_dir = "Capstone"

zip_path = f"./{base_dir}/KolektorSDD2.zip"
extract_path=f"./{base_dir}/"

def extract_zip(zip_path=zip_path, extract_path=extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed!")

def organize_files(base_dir="Capstone/"):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    train_images_dir = os.path.join(base_dir, "train_images")
    train_masks_dir = os.path.join(base_dir, "train_masks")
    test_images_dir = os.path.join(base_dir, "test_images")
    test_masks_dir = os.path.join(base_dir, "test_masks")

    for folder in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir]:
        os.makedirs(folder, exist_ok=True)

    def move_files(source_dir, image_dest, mask_dest):
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.isfile(file_path):
                if "_GT" in filename:
                    shutil.move(file_path, os.path.join(mask_dest, filename))
                else:
                    shutil.move(file_path, os.path.join(image_dest, filename))

    move_files(train_dir, train_images_dir, train_masks_dir)
    move_files(test_dir, test_images_dir, test_masks_dir)
    print("Files organized successfully!")

def create_csv(image_dir, mask_dir, output_csv):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    image_stems = {os.path.splitext(f)[0]: f for f in image_files}
    mask_stems = {os.path.splitext(f)[0]: f for f in mask_files}

    data = []
    for img_stem, img_file in image_stems.items():
        mask_file = mask_stems.get(img_stem + "_GT")
        if mask_file:
            data.append([os.path.join(mask_dir, mask_file), os.path.join(image_dir, img_file)])

    df = pd.DataFrame(data, columns=["masks", "images"])
    df.to_csv(output_csv, index=False)
    print(f"CSV saved: {output_csv}")