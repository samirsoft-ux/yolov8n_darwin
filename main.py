import cv2
from typing import List
import numpy as np
from ultralytics import YOLO
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.draw.color import ColorPalette
from yolox.tracker.byte_tracker import BYTETracker, STrack
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch

# URL de la cámara IP RTSP
rtsp_url = 'rtsp://admin:Hik12345@192.168.1.151:554/Streaming/Channels/101'

# Crear una instancia del modelo YOLOv8n
model = YOLO('/root/yolov8n_darwin/person.pt')

# Argumentos para ByteTrack
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# Inicializar ByteTrack
byte_tracker = BYTETracker(BYTETrackerArgs())

# Obtener información del video de la cámara RTSP
cap = cv2.VideoCapture(rtsp_url)

# Especifica la resolución deseada
desired_width = 640  # Cambiar por el ancho deseado
desired_height = 480  # Cambiar por la altura deseada

# Crear los anotadores de detecciones y línea
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
line_counter = LineCounter(start=(100, 400), end=(500, 400))  # Ajustar según tu caso
line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Función para convertir las detecciones en un formato compatible con ByteTrack
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)

def match_detections_with_tracks(detections: Detections, tracks: List[STrack]) -> np.ndarray:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))
    tracks_boxes = tracks2boxes(tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
    return tracker_ids

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame capturado al tamaño deseado
    resized_frame = cv2.resize(frame, (desired_width, desired_height))

    # Realizar la predicción con YOLOv8n en el frame redimensionado
    results = model(resized_frame)

    # Convertir las detecciones a formato de Supervision
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )

    # Actualizar ByteTrack con las detecciones actuales
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=resized_frame.shape,
        img_size=resized_frame.shape
    )

    # Asignar IDs de los rastreadores a las detecciones
    tracker_ids = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_ids)

    # Filtrar detecciones que no tienen un ID de seguimiento
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)

    # Actualizar el contador de línea (para verificar si las personas entran o salen)
    line_counter.update(detections=detections)

    # Anotar las cajas delimitadoras y el contador de línea
    frame_with_boxes = box_annotator.annotate(frame=resized_frame, detections=detections)
    line_annotator.annotate(frame=frame_with_boxes, line_counter=line_counter)

    # Mostrar el frame anotado
    cv2.imshow('YOLOv8 + ByteTrack - Conteo de Personas', frame_with_boxes)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
