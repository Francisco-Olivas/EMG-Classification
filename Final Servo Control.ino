#include <Servo.h>

Servo miServo, miServo2;
String entradaSerial = "";

void setup() {
  Serial.begin(9600);      // Comunicación serial
  miServo.attach(9);       // Conectar el servo
  miServo.write(90);       // Posición inicial
  miServo2.attach(8);       // Conectar el servo
  miServo.write(90);       // Posición inicial
}

void loop() {
  //Lectura del sensor EMG y envío a Python
  int emg = analogRead(A1);
  Serial.println(emg);
  delay(1);  // Aprox. 1000 Hz

  // Leer clase enviada desde Python
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      int clase = entradaSerial.toInt();  // Convierte texto a número

      // Controlar servos
      if (clase == 0) {
        miServo.write(10);  // Relajado
        miServo2.write(10);
      } else if (clase == 1) {
        miServo.write(150);  // Fuerte agarre
      } else if (clase == 2) {
        miServo.write(150);  // Fuerte agarre
        miServo2.write(150); 
      } 

      entradaSerial = "";  // Limpiar buffer
    } else {
      entradaSerial += c;  // Acumular caracteres
    }
  }
}
