import cv2

#step 1
def detect_faces(image, scaleFactor=1.1, minNeighbors=5):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
    return faces

def blur_face(image, face):
    (x, y, w, h) = face
    face_region = image[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
    image[y:y+h, x:x+w] = blurred_face
    return image

def pixelate_face(image, face, blocks=10):
    (x, y, w, h) = face
    face_region = image[y:y+h, x:x+w]
    temp = cv2.resize(face_region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = pixelated_face
    return image

#use
image = cv2.imread("C:/Users/HP/Desktop/TN10/photo.jpeg")
faces = detect_faces(image)

# create a copy
blurred_img = image.copy()
pixelated_img = image.copy()

for face in faces:
    blurred_img = blur_face(blurred_img, face)
    pixelated_img = pixelate_face(pixelated_img, face)

# save
cv2.imshow("Blurred", blurred_img)
cv2.imshow("Pixelated", pixelated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
