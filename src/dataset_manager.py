import os
from kaggle.api.kaggle_api_extended import KaggleApi


DATASETS = {
    "Massachusetts Buildings Dataset": "balraj98/massachusetts-buildings-dataset",
    "TuSimple Lane Detection Dataset": "manideep1108/tusimple",
}


def find_images_recursively(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file_name))

    return image_paths


def download_limited_images(dataset_name, max_images, output_dir="datasets"):
    if dataset_name not in DATASETS:
        raise ValueError("Unknown dataset selected.")

    dataset_id = DATASETS[dataset_name]

    api = KaggleApi()
    api.authenticate()

    dataset_output_dir = os.path.join(
        output_dir,
        dataset_name.lower().replace(" ", "_").replace("/", "_")
    )

    os.makedirs(dataset_output_dir, exist_ok=True)

    files = api.dataset_list_files(dataset_id).files

    image_files = [
        file.name for file in files
        if file.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    selected_files = image_files[:max_images]

    if not selected_files:
        raise ValueError("No image files found in this Kaggle dataset.")

    for file_name in selected_files:
        api.dataset_download_file(
            dataset=dataset_id,
            file_name=file_name,
            path=dataset_output_dir,
            quiet=False
        )

    return dataset_output_dir