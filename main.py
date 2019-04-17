import serial.tools.list_ports
import serial
import os
import time
ports = list(serial.tools.list_ports.comports())


#Getting the Com Port
for p in ports:
    #print(p[1]) #Uncomment this to see which ID your Arduino has.
    if "CH340" in p[1]: # Mine is not an original Arduino, and it has this ID. An original Arduino has "Arduino" as ID
                        # This should be changed acordignly.
        comPort=p[0]

ard = serial.Serial(comPort,9600,timeout=5)

while True:
    ardRead = ard.readline()
    input = ardRead.decode("utf-8")

    if "ON" in input:
        print("ON")
        os.startfile("run.bat")
        time.sleep(30)
    elif "OFF" in input:
        print("OFF")
        os.startfile("stop.bat")
        time.sleep(10)