import numpy as np
import dlib
import cv2
from ir_cam import IrCamera

if __name__ == '__main__':

    hogFaceDetector = dlib.get_frontal_face_detector()
    cv2.namedWindow('OUTPUT', cv2.WINDOW_NORMAL)
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    with IrCamera("data/conf.xml".encode('utf-8')) as cap_thermal:
        cap_visual = cv2.VideoCapture(2)
        while True:
            raw_thermal_data_from_cam, data_p, _ = cap_thermal.get_frame()
            ret, frame = cap_visual.read()

            # rawmax = np.max(data_p)

            raw_thermal_data_from_cam = raw_thermal_data_from_cam
            converted_to_float_thermal_data = np.float32(raw_thermal_data_from_cam)
            converted_to_float_thermal_data_in_3canals = cv2.cvtColor(converted_to_float_thermal_data,
                                                                      cv2.COLOR_GRAY2RGB)
            converted_to_float_thermal_data_in_3canals_resized = cv2.resize(converted_to_float_thermal_data_in_3canals,
                                                                            (480, 480), interpolation=cv2.INTER_AREA)
            # print(np.max(converted_to_float_thermal_data_in_3canals_resized))
            thermal_data = converted_to_float_thermal_data_in_3canals_resized.astype(np.uint8)

            start_point = (224, 115)
            end_point = (405, 305)

            frame = frame[115:305, 224:405]
            frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # faceRects = hogFaceDetector(frame, 0)

            # for faceRect in faceRects:

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x1, y1, w, h) in faces:
                x2 = x1 + w
                y2 = y1 + h
                y1 = int(y1 - (((y2 - y1) * 1.3) - (y2 - y1)))
                forehead = data_p[int(y1/6):int(y2/6), int(x1/6):int(x2/6)]
                if forehead.shape[0] > 0 and forehead.shape[1] > 0:

                    predict = np.max(forehead)
                    image = cv2.putText(thermal_data, str(predict)[1]+str(predict)[2]+"," + str(predict)[3], (x2, y2), font,
                                        1, (255, 255, 255), 1, cv2.LINE_AA)

                thermal_data = cv2.rectangle(thermal_data, (x1, y1), (x2, y2),
                                             (255, 255, 255), 2)

            predictAllArea = np.max(data_p)
            predict = predictAllArea
            image = cv2.putText(thermal_data,
                                "All Area Maximum: " + str(predict)[1] + str(predict)[2] + "," + str(predict)[3],
                                (30, 30), font,
                                1, (255, 255, 255), 1, cv2.LINE_AA)

            img_concate_Hori = np.concatenate((thermal_data * 2, frame), axis=1)
            cv2.imshow('OUTPUT', img_concate_Hori)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap_visual.release()
cv2.destroyAllWindows()
