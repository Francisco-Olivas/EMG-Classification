#include <Servo.h>

Servo miServo, miServo2, miServo3, miServo4, miServo5;
String entradaSerial = "";

void setup() {
  Serial.begin(9600);      // Comunicación serial
  miServo.attach(8);       // Conectar el servo
  miServo.write(100);       // Posición inicial
  miServo2.attach(9);       // Conectar el servo
  miServo2.write(100);       // Posición inicial
  miServo3.attach(10);       // Conectar el servo
  miServo3.write(100);       // Posición inicial
  miServo4.attach(11);       // Conectar el servo
  miServo4.write(40);       // Posición inicial
  miServo5.attach(12);       // Conectar el servo
  miServo5.write(40);       // Posición inicial
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
        miServo.write(40);       // Posición inicial
        miServo2.write(100);       // Posición inicial
        miServo3.write(40);       // Posición inicial
        miServo4.write(100);       // Posición inicial
        miServo5.write(120);       // Posición inicial
        delay(500);
      } else if (clase == 1) {
        miServo.write(20);  // Relajado
        miServo2.write(150);       // Posición c
        miServo3.write(150);       // Posición c
        miServo4.write(10);       // Posición c
        miServo5.write(40);       // Posición c
        delay(500);
      } else if (clase == 2) {
        miServo.write(150);  // Relajado
        miServo2.write(150);       // Posición c
        miServo3.write(150);       // Posición c
        miServo4.write(10);       // Posición c
        miServo5.write(40);       // Posición c
        delay(500);
      } 

      entradaSerial = "";  // Limpiar buffer
    } else {
      entradaSerial += c;  // Acumular caracteres
    }
  }
}