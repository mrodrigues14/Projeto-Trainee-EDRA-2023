import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
cap = cv2.VideoCapture(0)
 
while True:
	ret, frame = cap.read()
	detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
	corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

	if ids is not None:
		frame = cv2.aruco.drawDetectedMarkers(frame, corners)

		for i, id in enumerate(ids):
			c = corners[i][0]
			x = int((c[0][0] + c[2][0]) / 2)
			y = int((c[0][1] + c[2][1]) / 2)
			cv2.putText(frame, str(id[0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)		

	cv2.imshow('frame', frame)	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
