import cv2
import numpy as np
import glob

# 1. Priprema 3D tačaka (svaka tačka predstavlja ugao šahovske table u realnom prostoru)
chessboard_size = (9, 6)  # Dimenzije šahovske table (broj unutrašnjih uglova po redu i koloni)

# Generisanje koordinata za 3D tačke (realne tačke šahovske table u prostoru)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 2. Priprema za čuvanje 3D i 2D tačaka
obj_points = []  # 3D tačke u realnom prostoru
img_points = []  # 2D tačke u slici (detektovani uglovi šahovske table)

# 3. Učitavanje slika šahovske table
images = glob.glob('camera_cal\*.jpg')  # Putanja do foldera sa slikama

for fname in images:
    img = cv2.imread(fname)  # Učitavanje slike
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konverzija slike u sivu skaliranu

    # 4. Pronalaženje uglova šahovske table
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        obj_points.append(objp)  # Dodaj 3D tačke
        img_points.append(corners)  # Dodaj 2D tačke

        # Prikaz detektovanih uglova
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        #cv2.imshow('Chessboard', img)
        #cv2.waitKey(1000)

cv2.destroyAllWindows()

# 5. Kalibracija kamere
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 6. Ispis rezultata
print("Camera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)

#Ispravljanje slike 
img = cv2.imread('camera_cal\calibration1.jpg')
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
cv2.imshow('Distorted Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
