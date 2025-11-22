import cv2
from ultralytics import YOLO

# 1. Chargement du modèle entraîné
model_path = "runs/detect/train/weights/best.pt" 

# Si vous n'avez pas le fichier localement, remplacez par 'yolov5su.pt' pour tester le modèle de base
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Erreur: Impossible de charger le modèle à '{model_path}'.")
    print("Vérifiez le chemin. Utilisation du modèle de base pour l'exemple...")
    model = YOLO("yolov5su.pt")

# 2. Ouverture de la caméra
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

# Configuration de la fenêtre d'affichage
window_name = "Detection Geste Main - YOLO Live"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("Appuyez sur 'q' pour quitter la détection en direct.")

while True:
    # 3. Lire une image (frame) de la caméra
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture du flux vidéo.")
        break

    # 4. Faire la prédiction sur l'image actuelle
    results = model(frame, stream=True)

    # 5. Dessiner les résultats sur l'image
    for result in results:
        annotated_frame = result.plot()

    # 6. Afficher l'image annotée
    cv2.imshow(window_name, annotated_frame)

    # 7. Quitter si l'utilisateur appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()