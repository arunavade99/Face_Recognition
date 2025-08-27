# rebuild_db.py
import os
import argparse
import shutil
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# -------------------------------
# Config
# -------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, device=device)  # aligner
embed_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # embeddings

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# -------------------------------
# Utils
# -------------------------------
def is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMG_EXTS

def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / max(n, eps)

def list_dirs(path: str):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------------
# 1) Folder-level sync
# -------------------------------
def sync_person_folders(dataset_dir: str, aligned_dir: str):
    """Make aligned/ people folders match dataset/ exactly:
       - keep existing names
       - remove extras from aligned/
       - create new names missing in aligned/
    """
    ensure_dir(aligned_dir)

    dataset_people = set(list_dirs(dataset_dir))
    aligned_people = set(list_dirs(aligned_dir))

    # remove extra folders from aligned/
    extra = aligned_people - dataset_people
    for person in sorted(extra):
        path = os.path.join(aligned_dir, person)
        print(f"[SYNC] Removing extra folder from aligned/: {person}")
        shutil.rmtree(path, ignore_errors=True)

    # add new folders to aligned/
    missing = dataset_people - aligned_people
    for person in sorted(missing):
        path = os.path.join(aligned_dir, person)
        print(f"[SYNC] Adding new folder to aligned/: {person}")
        ensure_dir(path)

    # keep already-named folders (no action needed)
    kept = dataset_people & aligned_people
    if kept:
        print(f"[SYNC] Keeping existing folders: {sorted(kept)}")

# -------------------------------
# 2) Align images (file-level sync)
# -------------------------------
def align_images(dataset_dir: str, aligned_dir: str):
    """For each person:
       - remove aligned files that don’t exist in dataset
       - create aligned images for new files that exist in dataset but not in aligned
       - skip images already aligned
    """
    people = list_dirs(dataset_dir)
    for person in people:
        src_person_dir = os.path.join(dataset_dir, person)
        dst_person_dir = os.path.join(aligned_dir, person)
        ensure_dir(dst_person_dir)

        src_files = sorted([f for f in os.listdir(src_person_dir) if is_image(f)])
        dst_files = sorted([f for f in os.listdir(dst_person_dir) if is_image(f)])

        src_set = set(src_files)
        dst_set = set(dst_files)

        # remove extra aligned files
        for fname in sorted(dst_set - src_set):
            rm_path = os.path.join(dst_person_dir, fname)
            print(f"[ALIGN] Removing extra aligned file: {person}/{fname}")
            try:
                os.remove(rm_path)
            except OSError:
                pass

        # create aligned versions for new files
        for fname in sorted(src_set - dst_set):
            src_path = os.path.join(src_person_dir, fname)
            dst_path = os.path.join(dst_person_dir, fname)
            try:
                img = Image.open(src_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                print(f"[WARN] Bad image skipped: {src_path} ({e})")
                continue

            try:
                face = mtcnn(img, save_path=dst_path)
                if face is None:
                    print(f"[WARN] No face detected: {person}/{fname}")
                    # ensure no half-written file remains
                    if os.path.exists(dst_path):
                        try: os.remove(dst_path)
                        except OSError: pass
                else:
                    print(f"[ALIGN] Saved: {person}/{fname}")
            except Exception as e:
                print(f"[WARN] Align failed for {person}/{fname}: {e}")
                if os.path.exists(dst_path):
                    try: os.remove(dst_path)
                    except OSError: pass

        # already aligned files are left untouched

# -------------------------------
# 3) Update embeddings DB
# -------------------------------
def get_embedding(img_path: str) -> np.ndarray:
    try:
        img = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"[WARN] Bad aligned image skipped: {img_path} ({e})")
        return None
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_model(img_tensor).cpu().numpy().reshape(-1)
    return l2_normalize(emb)

def compute_person_embedding(aligned_person_dir: str) -> np.ndarray | None:
    embs = []
    for fname in sorted(os.listdir(aligned_person_dir)):
        if not is_image(fname):
            continue
        path = os.path.join(aligned_person_dir, fname)
        emb = get_embedding(path)
        if emb is not None:
            embs.append(emb)
    if not embs:
        return None
    mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(mean_emb)

def update_db_from_aligned(aligned_dir: str, out_npz: str):
    # load existing DB if present
    if os.path.exists(out_npz):
        npz = np.load(out_npz, allow_pickle=True)
        embedding_db = {k: npz[k] for k in npz.files}
        print(f"[DB] Loaded existing DB: {out_npz}")
    else:
        embedding_db = {}
        print(f"[DB] Creating new DB: {out_npz}")

    current_people = set(list_dirs(aligned_dir))
    # remove people absent from aligned/
    for name in list(embedding_db.keys()):
        if name not in current_people:
            print(f"[DB] Removing {name} from DB (no aligned folder)")
            del embedding_db[name]

    # add/update people present in aligned/
    for person in sorted(current_people):
        person_dir = os.path.join(aligned_dir, person)
        emb = compute_person_embedding(person_dir)
        if emb is None:
            print(f"[DB] No usable embeddings for {person} (folder may be empty). Skipped.")
            # If previously existed, we keep old until folder gone.
            continue
        embedding_db[person] = emb.astype(np.float32)
        print(f"[DB] Updated embedding for: {person}")

    # save
    np.savez(out_npz, **embedding_db)
    print(f"[DB] Saved → {out_npz}")
    print(f"[DB] Persons: {sorted(embedding_db.keys())}")

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sync folders → align → update embeddings.npz")
    parser.add_argument("--dataset_dir", required=True, help="Path to raw dataset (person-wise folders)")
    parser.add_argument("--aligned_dir", default="aligned", help="Path to aligned dataset folder")
    parser.add_argument("--out_npz", default="embeddings.npz", help="Output embeddings DB path")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {args.dataset_dir}")

    ensure_dir(args.aligned_dir)

    print("\n[STEP 1] Sync person folders (aligned ⇄ dataset)")
    sync_person_folders(args.dataset_dir, args.aligned_dir)

    print("\n[STEP 2] Align images (remove extras, add new, keep existing)")
    align_images(args.dataset_dir, args.aligned_dir)

    print("\n[STEP 3] Update embeddings database")
    update_db_from_aligned(args.aligned_dir, args.out_npz)

    print("\n Done.")

if __name__ == "__main__":
    main()
