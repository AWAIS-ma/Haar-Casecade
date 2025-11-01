import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image, save_path=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if it doesn't exist
        cv2.imwrite(save_path, image)
        print(f"Detected image saved as: {save_path}")

def choose_image_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    root.destroy()
    return file_path

def live_camera_face_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access the camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Live Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Menu options
print("Select an option:")
print("1: Detect faces in an image file (PNG or JPG)")
print("2: Live face detection using the camera")
choice = input("Enter 1 or 2: ")

if choice == '1':
    file_path = choose_image_file()
    if file_path:
        image = cv2.imread(file_path)
        base_name = os.path.basename(file_path)  # Get only the file name
        name, ext = os.path.splitext(base_name)
        save_path = f"output_images/{name}_detected{ext}"
        detect_faces(image, save_path=save_path)
    else:
        print("No file selected.")
elif choice == '2':
    live_camera_face_detection()
else:
    print("Invalid choice. Please select 1 or 2.")
