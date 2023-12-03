# Importamos las librerias
from deepface import DeepFace
import cv2
import mediapipe as mp

# Declaramos la deteccion de rostros
detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence=0.8, model_selection=0)
# Dibujo
dibujorostro = mp.solutions.drawing_utils

# Leemos imagen de fondo para mostrar resultados
img_fondo = cv2.imread("edad.png")
img_fondo = cv2.resize(img_fondo, (0, 0), None, 0.18, 0.18)
ani, ali, c = img_fondo.shape

# Realizamos VideoCaptura
cap = cv2.VideoCapture(0)  # Usamos el índice 0 para la cámara predeterminada

# Empezamos
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()
    if not ret:
        break  # Salir si no se puede capturar el fotograma

    # Correccion de color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesamos la detección de rostros
    resrostros = rostros.process(rgb)
    if resrostros.detections is not None:
        # Registramos cada rostro detectado
        for rostro in resrostros.detections:
            box = rostro.location_data.relative_bounding_box
            al, an, c = frame.shape
            xi, yi, w, h = int(box.xmin * an), int(box.ymin * al), int(box.width * an), int(box.height * al)
            xf, yf = xi + w, yi + h

            # Dibujamos el rectángulo del rostro detectado
            cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 255, 0), 1)
            frame[10:ani + 10, 10:ali + 10] = img_fondo  # Colocamos la imagen de fondo

            # Información
            info = DeepFace.analyze(rgb, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

            # Obtener información para mostrar
            gen = "Hombre" if info['gender'] == 'Man' else "Mujer"
            emociones = info['dominant_emotion']
            race = info['dominant_race']
            edad = info['age']

            # Traducción de emociones y razas
            emociones_es = {'angry': 'enojado', 'disgust': 'disgustado', 'fear': 'miedoso',
                            'happy': 'feliz', 'sad': 'triste', 'surprise': 'sorprendido', 'neutral': 'neutral'}
            race_es = {'asian': 'asiatico', 'indian': 'indio', 'black': 'negro', 'white': 'blanco',
                       'middle eastern': 'oriente medio', 'latino hispanic': 'latino'}

            emociones = emociones_es.get(emociones, emociones)
            race = race_es.get(race, race)

            # Mostramos la información en el fotograma con la fuente Arial Unicode MS
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, gen, (65, 50), font, 1, (0, 0, 0), 2)
            frame = cv2.putText(frame, f"{edad}", (75, 90), font, 1, (0, 0, 0), 2)
            frame = cv2.putText(frame, emociones, (75, 135), font, 1, (0, 0, 0), 2)
            frame = cv2.putText(frame, race, (75, 180), font, 1, (0, 0, 0), 2)

    # Mostramos los fotogramas
    cv2.imshow("Deteccion de Edad", frame)

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break  # Salir con la tecla Esc

cv2.destroyAllWindows()
cap.release()
