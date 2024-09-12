Pasos para replicar el ambiente:

- conda create --name ambiente python=3..8.19

- conda activate ambiente

- pip install numpy==1.19.5

- conda install --file paquetes_conda.txt

- pip install -r paquetes_pip.txt

- git clone https://github.com/ifzhang/ByteTrack.git (si esta en el repositorio ya no es necesario)

- cd ByteTrack

- python setup.py -q develop

Notas a tener en cuenta:
el archivo backup original de mi entorno que esta funcionando pero esta con una versión de python no soportada

- test: es el nombre del entorno
- original_entorno.yml: es el archivo que guarda todas las dependencias


- backup_entorno.yml: es donde he modificado la versión de python a la 3.9

------------------------------------------------------------------------------------------------------------------

Te explico el significado de cada uno de los argumentos dentro de la clase BYTETrackerArgs, para que puedas ajustarlos de acuerdo a las necesidades de tu caso de uso:

1. track_thresh: float = 0.25
Descripción: Este es el umbral de confianza mínimo para que una detección sea considerada válida para el seguimiento. Es decir, si la confianza de una detección es inferior a este valor, no se le asignará un identificador de seguimiento (ID).
Cómo modificarlo: Si estás trabajando en un entorno donde las detecciones son poco confiables (por ejemplo, con poca iluminación o ángulos difíciles), puedes disminuir este valor para que más objetos sean rastreados, aunque con más ruido. Si, por el contrario, solo quieres rastrear objetos con alta precisión, aumenta este valor.
Ejemplo: Para situaciones de alta precisión, podrías configurarlo en 0.5 o incluso 0.7.
2. track_buffer: int = 30
Descripción: Es el número de cuadros (frames) que el rastreador "recordará" una detección antes de considerarla perdida o desaparecida. Si el objeto desaparece por menos de este número de frames, se mantendrá su ID de rastreo. Si desaparece por más tiempo, se le asignará un nuevo ID si vuelve a aparecer.
Cómo modificarlo: Si esperas oclusiones o que los objetos salgan temporalmente de la vista (como personas entrando o saliendo de una puerta), puedes aumentar este valor para mantener el seguimiento durante más tiempo. Si estás en un entorno muy dinámico con muchas personas entrando y saliendo rápidamente, podrías reducir este valor para asegurar que los IDs se reasignen con mayor rapidez.
Ejemplo: En un entorno de oficina con oclusiones ocasionales, podrías aumentarlo a 60 o más.
3. match_thresh: float = 0.8
Descripción: Este es el umbral de coincidencia (IoU, Intersection over Union) entre una detección y una predicción anterior para considerarlas el mismo objeto. Cuanto más alto sea este valor, más estricta será la coincidencia entre la detección actual y el estado previo del objeto.
Cómo modificarlo: Si los objetos en tu entorno no cambian mucho de forma o tamaño (por ejemplo, personas caminando en línea recta), puedes aumentar este valor para tener un seguimiento más preciso. Si hay cambios rápidos en la forma o la orientación de los objetos (por ejemplo, personas que corren o giran bruscamente), podrías disminuir este valor para permitir coincidencias más laxas.
Ejemplo: En un entorno donde el movimiento es más errático, podrías reducirlo a 0.5.
4. aspect_ratio_thresh: float = 3.0
Descripción: Este es el umbral de relación de aspecto. Si el rastreador detecta que la relación de aspecto de un objeto cambia más allá de este valor, puede dejar de rastrearlo. Esto previene que un objeto con cambios extremos de forma siga siendo rastreado incorrectamente.
Cómo modificarlo: Si estás rastreando personas (cuya relación de aspecto no debería cambiar drásticamente), podrías reducir este valor para ser más estricto. Si, por otro lado, los objetos pueden tener cambios significativos en su relación de aspecto, como personas que se agachan o se mueven de diferentes maneras, aumenta este valor.
Ejemplo: Para rastrear personas, podrías reducirlo a 2.0 para mayor precisión, o aumentarlo si hay mucha variabilidad en las posturas.
5. min_box_area: float = 1.0
Descripción: Es el área mínima de las cajas delimitadoras (bounding boxes) que serán rastreadas. Las cajas más pequeñas que este valor serán ignoradas.
Cómo modificarlo: Si estás rastreando personas u objetos grandes, puedes aumentar este valor para ignorar objetos muy pequeños o detecciones erróneas. Si te interesa rastrear pequeños objetos o detalles (por ejemplo, en un entorno de vigilancia), podrías disminuir este valor.
Ejemplo: Si no te interesan objetos pequeños, podrías aumentarlo a 20.0 o más, dependiendo del tamaño mínimo que te interese rastrear.
6. mot20: bool = False
Descripción: Este es un parámetro booleano que indica si estás utilizando el modelo de rastreo optimizado para el conjunto de datos MOT20, que está enfocado en escenarios de seguimiento denso, como multitudes.
Cómo modificarlo: Si tu caso de uso implica un entorno muy denso, como un espacio con muchas personas, podrías establecer este parámetro en True. Si no, lo más adecuado es dejarlo en False.
Ejemplo: Si estás trabajando en una oficina o espacio donde la densidad de personas es baja o moderada, puedes dejarlo en False.
Resumen de ajustes posibles para tu caso de uso:
Si estás trabajando en una oficina con oclusiones y quieres evitar perder el seguimiento de personas temporalmente:

track_buffer: Aumentar a 60.
track_thresh: Mantener en 0.25 o ajustarlo a 0.3 para filtrar detecciones de baja confianza.
Si hay cambios rápidos en el movimiento de las personas (gente corriendo o moviéndose rápidamente):

match_thresh: Bajar a 0.5.
Si hay objetos muy pequeños en el fondo que no te interesan:

min_box_area: Aumentar a 20.0 o más para filtrar objetos pequeños.
Estos parámetros te permitirán ajustar ByteTrack para adaptarlo a la dinámica de tu entorno.



PROXIMOS FEATURES
- LABELS MODIFICADOS SEGÚN EL VIDEO
- ENTENDER LA LINEA DONDE SE ESTA DIBUJANDO
- ENTENDER COMO SE ESTA CONTANDO LAS PERSONAS
