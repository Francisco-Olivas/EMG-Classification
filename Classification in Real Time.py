import serial
import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt

# CONFIGURACIÓN
VENTANA = 100
TRASLAPE = 50
FS = 1000  
PORT = 'COM12' 
BAUDRATE = 9600
MODELO_PATH = 'modelo_svm.pkl'

# FILTRO PASA BANDA
def butter_bandpass_filter(data, lowcut=20.0, highcut=450.0, fs=1000.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# EXTRACCIÓN DE CARACTERÍSTICAS
def extraer_caracteristicas(ventana_filtrada):
    rectificada = np.abs(ventana_filtrada)
    rms = np.sqrt(np.mean(np.square(rectificada)))
    media = np.mean(rectificada)
    varianza = np.var(rectificada)
    maximo = np.max(rectificada)
    minimo = np.min(rectificada)
    rango = maximo - minimo
    return [rms, media, varianza, maximo, minimo, rango]

# CONECTAR Y CARGAR MODELO
print("Cargando modelo...")
modelo = joblib.load(MODELO_PATH)

print(f"Conectando a {PORT} a {BAUDRATE} baud...")
arduino = serial.Serial(PORT, BAUDRATE, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

print("Clasificación en tiempo real iniciada...\n")

# CLASIFICACIÓN
buffer_ventana = []
clase_anterior = None
contador_clase = 0  # Contador para verificar repeticiones de clase

try:
    while True:
        # Leer datos del puerto serial del Arduino (sensor EMG)
        linea = arduino.readline().decode('utf-8', errors='ignore').strip()
        if linea.isdigit():
            valor = int(linea)
            buffer_ventana.append(valor)

            if len(buffer_ventana) == VENTANA:
                ventana_np = np.array(buffer_ventana, dtype=float)

                # Filtrado de la señal EMG
                try:
                    ventana_filtrada = butter_bandpass_filter(ventana_np, fs=FS)
                except Exception as fe:
                    print(f"[Filtro] Error: {fe}")
                    buffer_ventana = buffer_ventana[int(VENTANA / 2):]
                    continue

                # Extracción de características
                caracteristicas = extraer_caracteristicas(ventana_filtrada)
                entrada = np.array([caracteristicas])

                # Predicción de clase
                pred = modelo.predict(entrada)
                clase = pred[0]
                print(f"Clase estimada: {clase}")

                # Verificar si la clase actual es la misma que la anterior
                if clase == clase_anterior:
                    contador_clase += 1
                else:
                    contador_clase = 1  # Reiniciar contador

                # Solo actualizar el servo si la clase se repite tres veces seguidas
                if contador_clase >= 2:
                    # Enviar clase al Arduino
                    arduino.write(f"{clase}\n".encode())
                    print(f"Clase enviada al Arduino: {clase}")
                    # Reiniciar el contador después de enviar
                    contador_clase = 0

                # Aplicar traslape
                buffer_ventana = buffer_ventana[TRASLAPE:]

                # Actualizar la clase anterior
                clase_anterior = clase

except KeyboardInterrupt:
    print("Interrumpido por el usuario.")
except Exception as e:
    print(f"Error general: {e}")
finally:
    arduino.close()
    print("Puerto serial cerrado.")
