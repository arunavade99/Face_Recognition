# rebuild_db.py
import os
import argparse
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Helper functions
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_model(img_tensor).cpu().numpy().flatten()
    return emb

def process_person(person_dir):
    embeddings = []
    for file in os.listdir(person_dir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(person_dir, file)
            emb = get_embedding(path)
            embeddings.append(emb)
    if embeddings:
        return np.mean(embeddings, axis=0)  # average embedding per person
    else:
        return None
    
# Main
def main(args):
    # Load existing DB if exists
    if os.path.exists(args.out_npz):
        print(f"[INFO] Loading existing DB: {args.out_npz}")
        db = np.load(args.out_npz, allow_pickle=True)
        embedding_db = {name: db[name] for name in db.files}
    else:
        print("[INFO] No existing DB found. Creating new one...")
        embedding_db = {}

    # Go through dataset folders
    for person_name in os.listdir(args.aligned_dir):
        person_dir = os.path.join(args.aligned_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        emb = process_person(person_dir)
        if emb is not None:
            print(f"[INFO] Updating embedding for: {person_name}")
            embedding_db[person_name] = emb

    # Save updated DB
    np.savez(args.out_npz, **embedding_db)
    print(f"[INFO] Database saved at {args.out_npz}. Current persons: {list(embedding_db.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_dir", type=str, required=True,
                        help="Path to aligned dataset folder")
    parser.add_argument("--out_npz", type=str, default="embeddings.npz",
                        help="Output embeddings database file")
    args = parser.parse_args()
    main(args)
