import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load FaceNet
embed_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)

# Load embedding DB
db = np.load("embeddings.npz")
embedding_db = {name: db[name] for name in db.files}
print("Database loaded:", list(embedding_db.keys()))

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect multiple faces
    boxes, probs = mtcnn.detect(img)
    faces = mtcnn(img)

    if faces is not None and boxes is not None:
        with torch.no_grad():
            embeddings = embed_model(faces.to(device)).cpu().numpy()

        for i, emb in enumerate(embeddings):
            box = boxes[i]
            (x1, y1, x2, y2) = [int(v) for v in box]

            # Compare with DB
            best_match, min_dist = None, float("inf")
            for name, db_emb in embedding_db.items():
                dist = np.linalg.norm(emb - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_match = name

            # Recognition threshold
            if min_dist < 0.9:
                text = f"{best_match} ({min_dist:.2f})"
            else:
                text = "Unknown"

            # Draw bounding box + text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
