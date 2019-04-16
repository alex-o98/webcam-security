# About

This repository contains a home webcam security system with face recognition and motion detection built-in. The security system starts/shuts after another python script listens to an Arduino COM which sends a Serial input after a button has been pressed (On/Off). It then runs or stops face_recon.py (Based on the input). 

face_recon.py works by detecting movement or face in a camera. If any movement has been registered, it saves a 10 seconds recording of a file from the last movement appearance and sends it to a remote server to have a back up of the recording (and also to have remote access to the file). If a movement has been registered and meanwhile (before saving the recording) a known face has been detected, it forgets about the "10 seconds" rule for 60 seconds, so it won't save the recording. 

# Known issues / Things to be changed

The program has been built for Windows, with the intention to be transfered on a local server running linux, and the files run.bat and stop.bat should be changed to execute or stop face_recon.py.

Because face_recon.py has been designed only for a camera, if it is implemented in an advanced security system, face_recon.py should be changed so that it works on multiple cameras at once (Maybe even have it read every single camera available and adjust the program to that). 

Even though it requires a lot of power to run the face detection, in a real environment (depending on how the cameras are pointed) it is unlikely to detect faces in multiple cameras (in case of a break-in). That is not a problem for the motion detection algorithm as it is relatively light weight.

# Future updates
  - Have it save locally new unkown faces and asign a value to that person.
  - Make it work on multiple cameras without having to modify the code every time the code is introduced.
  - Create an option to recieve push-notifications (on Android)
  - Create a web app/mobile app to configure settings / start or stop the program,etc.
  - And more.

# Credits

[habrman](https://github.com/habrman) for creating the Face Recognition algorithm. Link to the repository [here](https://github.com/habrman/FaceRecognition).
