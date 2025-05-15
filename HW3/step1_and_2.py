import cv2
from deepface import DeepFace
import os

# ---------------------
# STEP 1: Dé-identification
# ---------------------

def detect_faces(image, scaleFactor=1.1, minNeighbors=5):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
    return faces

def blur_face(image, face):
    (x, y, w, h) = face
    face_region = image[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face_region, (15, 15), 5)  # moins extrême
    image[y:y+h, x:x+w] = blurred_face
    return image

def pixelate_face(image, face, blocks=40):  # plus fin que blocks=10
    (x, y, w, h) = face
    face_region = image[y:y+h, x:x+w]
    temp = cv2.resize(face_region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = pixelated_face
    return image

# original pitcture
original_path = "C:/Users/HP/Desktop/TN10/photo_cv.jpeg"
image = cv2.imread(original_path)
if image is None:
    print("Erreur : impossible de charger l'image.")
    exit()

#face detection
faces = detect_faces(image)
if len(faces) == 0:
    print("Aucun visage détecté.")
    exit()

# pitcture copy
blurred_img = image.copy()
pixelated_img = image.copy()

# traitements
for face in faces:
    blurred_img = blur_face(blurred_img, face)
    pixelated_img = pixelate_face(pixelated_img, face)

# save
blurred_path = "C:/Users/HP/Desktop/TN10/photo_cv_blurred.jpg"
pixelated_path = "C:/Users/HP/Desktop/TN10/photo_cv_pixelated.jpg"

cv2.imwrite(blurred_path, blurred_img)
cv2.imwrite(pixelated_path, pixelated_img)

# show
cv2.imshow("Original", image)
cv2.imshow("Blurred", blurred_img)
cv2.imshow("Pixelated", pixelated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------
# STEP 2: recognize the original face
# ---------------------

# files
base_dir = "C:/Users/HP/Desktop/TN10"    #file with picture
photo_dir = os.path.join(base_dir, "photo")  # file with test pictures and the original one

# blurred and pixelated picture to test
deidentified_images = {
    "photo_cv_blurred.jpg": os.path.join(base_dir, "photo_cv_blurred.jpg"),  #blurred picture
    "photo_cv_pixelated.jpg": os.path.join(base_dir, "photo_cv_pixelated.jpg")   #pixelated picture
}

# original picture to compare
original_photos = [
    f for f in os.listdir(photo_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print("\n--- TEST OF MATCHING ---\n")

for label, deid_path in deidentified_images.items():
    print(f"Analyse of : {label}")
    match_found = False

    for filename in original_photos:
        original_path = os.path.join(photo_dir, filename)
        try:
            result = DeepFace.verify(deid_path, original_path, enforce_detection=False, model_name="Facenet")
            if result["verified"]:
                print(f"Matching find with {filename}")
                match_found = True
                break  # stop after we find one
        except Exception as e:
            print(f"Error with {filename} : {e}")

    if not match_found:
        print("No matching\n")
    else:
        print()
