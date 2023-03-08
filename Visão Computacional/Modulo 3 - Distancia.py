import cv2
from cv2 import aruco
import numpy as np

calib_data_path = "calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 8

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters()
cap = cv2.VideoCapture(0)

trigger = 0

while True:
    ret, frame = cap.read()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    marker_corners, marker_IDs, reject = detector.detectMarkers(frame)
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, cam_mat, dist_coef)
        totat_markers = range(0,marker_IDs.size)
        for ids, corners,i in zip(marker_IDs, marker_corners, totat_markers):
            cv2.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            #meker local
            poit = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 3, 2)
            cv2.putText(
                frame,
                f"Id: {ids[0]} Distancia: {round(tVec[i][0][2])}",
                top_right,
                cv2.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"X: {round(tVec[i][0][0],1)} Y: {round(tVec[i][0][1],1)}",
                bottom_left,
                cv2.FONT_HERSHEY_PLAIN,
                1.3,
                (0,0,255),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
