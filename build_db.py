# build_db.py
import os
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FaceNet
embed_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)

dataset_dir = "aligned/"
embedding_db = {}

# Build database
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    embeddings = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = Image.open(img_path)

        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                emb = embed_model(face.unsqueeze(0).to(device)).cpu().numpy()
            embeddings.append(emb)

    if embeddings:
        embedding_db[person_name] = np.mean(embeddings, axis=0)

# Save database
np.savez("embeddings.npz", **embedding_db)
print("✅ Embedding database saved as embeddings.npz")
