import datetime
from ftplib import FTP
import sys
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
from scipy import misc
import tensorflow as tf
import numpy as np
import detect_and_align
import argparse
import cv2
import os
import time
import pathlib


# DEF START:
# ./models/20170512-110547.pb ./ids/ camera

class IdData():
    """Keeps track of known identities and calculates id matches"""

    def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder,
                 phase_train_placeholder, distance_treshold):
        print('Loading known identities: ', end='')
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []

        image_paths = []
        ids = os.listdir(os.path.expanduser(id_folder))
        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)

            image_paths = image_paths + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        for i in image_paths:
            path = os.path.dirname(i)

        print('Found %d images in id folder' % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = misc.imread(os.path.expanduser(image_path), mode='RGB')
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print("Warning: Found multiple faces in id image: %s" % image_path +
                      "\nMake sure to only have one face in the id images. " +
                      "If that's the case then it's a false positive detection and" +
                      " you can solve it by increasing the thresolds of the cascade network")
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split('/')[-1] for path in id_image_paths]
        print('Distance matrix:\n{:20}'.format(''), end='')
        [print('{:20}'.format(name), end='') for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print('\n{:20}'.format(path), end='')
            for distance in distance_row:
                print('{:20}'.format('%0.3f' % distance), end='')
        print()

    def find_matching_ids(self, embs):
        matching_ids = []
        matching_distances = []
        distance_matrix = pairwise_distances(embs, self.embeddings)
        for distance_row in distance_matrix:
            min_index = np.argmin(distance_row)
            if distance_row[min_index] < self.distance_treshold:
                matching_ids.append(self.id_names[min_index])
                matching_distances.append(distance_row[min_index])
            else:
                matching_ids.append(None)
                matching_distances.append(None)
        return matching_ids, matching_distances


def load_model(model):
    model_exp = "models\\20170512-110547.pb"

    print('Loading model filename: %s' % model_exp)
    with gfile.FastGFile(model_exp, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def check_dir(dir, ftp_conn):
    filelist = []
    ftp_conn.retrlines('LIST', filelist.append)
    found = False

    for f in filelist:
        if f.split()[-1] == dir and f.lower().startswith('d'):
            found = True

    if not found:
        ftp_conn.mkd(dir)
    ftp_conn.cwd(dir)


def chdir(ftp_path, ftp_conn):
    dirs = [d for d in ftp_path.split('/') if d != '']
    for p in dirs:
        check_dir(p, ftp_conn)


def main(args):
    if False:
        print("Start input")
    else:
        with tf.Graph().as_default():
            with tf.Session() as sess:

                # Setup models
                mtcnn = detect_and_align.create_mtcnn(sess, None)
                load_model("models\\20170512-110547.pb")
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # Load anchor IDs
                id_data = IdData("ids", mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder,
                                 args.threshold)

                # print(len(args.id_folder[0]))

                cap = cv2.VideoCapture(0)

                det_time = time.time()

                kVids = 0
                first_frame = None

                save_rec = False
                init_out = False

                save_rec_time = time.time()

                fourcc = cv2.VideoWriter_fourcc(*'DIVX')

                while (True):
                    _, frame = cap.read()
                    objdet = frame

                    # Locate faces and landmarks in frame
                    face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)

                    if save_rec == True:
                        if init_out == False:
                            dt = datetime.datetime.now().strftime("%d %H-%M-%S.%f")[:-3]
                            fileToSend = dt
                            dmonth = datetime.datetime.now().strftime("%b")
                            dyear = datetime.datetime.now().strftime("%Y")

                            pathlib.Path('detection\\vids\\' + dyear + '\\' + dmonth).mkdir(parents=True, exist_ok=True)

                            out = cv2.VideoWriter('detection\\vids\\{2}\\{0}\\{1}.mkv'.format(dmonth, dt, dyear),
                                                  fourcc, fps=10,
                                                  frameSize=(640, 480))
                            init_out = True

                        out.write(frame)

                        if time.time() - save_rec_time > 15 and known_face == False:
                            save_rec = False
                            out.release()
                            kVids += 1
                            file = open('detection\\vids\\{2}\\{0}\\{1}.mkv'.format(dmonth, dt, dyear), 'rb')

                            folderName = dyear + '/' + dmonth
                            ftp = FTP('TO CHANGE')  # To change
                            ftp.login('TO CHANGE', 'TO CHANGE')  # To change
                            # Checking if file exists
                            chdir(folderName, ftp)

                            ftp.storbinary("STOR {0}.mkv".format(fileToSend), file)

                            file.close()
                            ftp.close()
                            init_out = False

                        elif known_face:
                            save_rec = False

                            out.release()

                            # Adding an extra step to be sure the file has been saved.
                            if os.path.exists('detection\\vids\\{0}\\{1}\\{2}.mkv'.format(dyear, dmonth, dt)):
                                os.remove('detection\\vids\\{0}\\{1}\\{2}.mkv'.format(dyear, dmonth, dt))
                            known_face_time = time.time()
                            init_out = False
                            known_face = False

                    if len(face_patches) > 0:
                        face_patches = np.stack(face_patches)
                        feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                        embs = sess.run(embeddings, feed_dict=feed_dict)

                        matching_ids, matching_distances = id_data.find_matching_ids(embs)

                        for bb, landmark, matching_id, dist in zip(padded_bounding_boxes, landmarks, matching_ids,
                                                                   matching_distances):
                            if matching_id is None:
                                matching_id = 'Unknown'

                                # Do not start the recording unless at least 60 seconds has been passed since a known
                                # face has been detected
                                if time.time() - known_face_time > 60:
                                    save_rec_time = time.time()
                                    save_rec = True
                            else:
                                known_face = True
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, matching_id, (bb[0], bb[3]), font, 1, (255, 255, 255), 4, cv2.LINE_AA)
                            cv2.putText(frame, matching_id, (bb[0], bb[3]), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
                            # how_landmarks:
                            for j in range(5):
                                size = 1
                                top_left = (int(landmark[j]) - size, int(landmark[j + 5]) - size)
                                bottom_right = (int(landmark[j]) + size, int(landmark[j + 5]) + size)
                                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)

                    else:
                        # Object detection
                        # Works only if no face detected.
                        status = -1
                        grayImg = cv2.cvtColor(objdet, cv2.COLOR_BGR2GRAY)
                        grayImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
                        if first_frame is None:
                            first_frame = grayImg
                        deltaFrame = cv2.absdiff(first_frame, grayImg)
                        threshFrame = cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY)[1]
                        threshFrame = cv2.dilate(threshFrame, None, iterations=2)
                        contours, hierachy = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if cv2.contourArea(contour) < 10000:
                                # excluding too small contours. Set 10000 (100x100 pixels) for objects close to camera
                                continue

                            save_rec = True
                            save_rec_time = time.time()

                            status = 1
                            # obtain the corresponding bounding rectangle of our detected contour
                            (x, y, w, h) = cv2.boundingRect(contour)

                            # superimpose a rectangle on the identified contour in our original colour image
                            # (x,y) is the top left corner, (x+w, y+h) is the bottom right corner
                            # (0,255,0) is colour green and 3 is the thickness of the rectangle edges
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        first_frame = grayImg

                    cv2.imshow('frame', frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float,
                        help='Distance threshold defining an id match', default=1.2)
    main(parser.parse_args())
