import cv2
from ultralytics import YOLO
import threading

class VideoCaptureThread(threading.Thread):
    def _init_(self, url):
        super()._init_()
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def stop(self):
        self.running = False
        self.cap.release()

# URL RTSP de la cámara Hikvision
rtsp_url = 'rtsp://admin:Hik12345@192.168.1.151:554/Streaming/Channels/101'

# Cargar el modelo entrenado YOLOv8
model = YOLO('C:/Users/jhonatan.chocce/Documents/proyectoyolo/models/person.pt')

# Iniciar el hilo de captura de video
video_thread = VideoCaptureThread(rtsp_url)
video_thread.start()

# create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)


while True:
    # Leer el fotograma del hilo de captura
    frame = video_thread.frame
    if frame is None:
        print("Error: No se puede leer el fotograma de la cámara")
        continue

    # Realizar la detección de objetos con el modelo entrenado en YOLOv8
    results = model.predict(frame)  # Realizar la detección

    # Dibujar las detecciones en el fotograma
    annotated_frame = results[0].plot()  # Anotar los resultados en el fotograma

    # Mostrar el fotograma anotado
    cv2.imshow('Detección de Personas con YOLOv8', annotated_frame)

    # Salir del bucle si se presiona la tecla 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:  # Código ASCII para 'Esc'
        break

# Detener el hilo de captura de video y liberar los recursos
video_thread.stop()
cv2.destroyAllWindows()