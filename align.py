import os
from PIL import Image
from facenet_pytorch import MTCNN

dataset_dir = "dataset/"
aligned_dir = "aligned/"

os.makedirs(aligned_dir, exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20)

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    save_dir = os.path.join(aligned_dir, person_name)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = Image.open(img_path)

        # Detect & align
        face = mtcnn(img, save_path=os.path.join(save_dir, img_name))
