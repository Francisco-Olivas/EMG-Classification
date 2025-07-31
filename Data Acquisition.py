import serial
import csv
import datetime

arduino = serial.Serial('COM12', 9600, timeout=1)

with open('contraccion3.csv', mode='w', newline='') as archivo:
    escritor_csv = csv.writer(archivo)
    escritor_csv.writerow(["Timestamp", "Valor Analogico"])  # Escribe el encabezado

    try:
        while True:
            datos = arduino.readline().decode().strip()  # Lee y decodifica datos
            if datos:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Obtiene el timestamp
                print(f"{timestamp}, {datos}")  # Muestra el timestamp junto con los datos en consola
                escritor_csv.writerow([timestamp, datos])  # Guarda en el CSV
            
    except KeyboardInterrupt:
        print("\n Cerrando conexión...")
        arduino.close()  #cierra en caso de interrupción manual
