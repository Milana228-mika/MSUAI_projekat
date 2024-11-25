import cv2
import numpy as np
import glob

chessboard_corners = (9, 6)  # Broj unutrasnjih uglova sahovske table(S obzirom da imamo kvadrata 10x7, uglova imamo 10x6)

# Najpre napravimo trodimenzijalnu matricu 54 elementa ispunjenu nulama, a zatim zamenimo to sa koordinatama uglova, Z osa ostaje na 0
matrix = np.zeros((chessboard_corners[0] * chessboard_corners[1], 3), np.float32)
matrix[:, :2] = np.mgrid[0:chessboard_corners[0], 0:chessboard_corners[1]].T.reshape(-1, 2)

matrix_points = []  # 3D tačke sahovske table 
image_points = []  # koordinate uglova detektovanih na slikama

images = glob.glob('camera_cal\*.jpg')  # Putanja do foldera sa slikama, uzimamo sve slike iz foldera

for image in images:
    img = cv2.imread(image)  # Učitavanje slike
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konertujemo sliku u sivu, kako bi se detektovale ivice na slici spram svetline

    # Pronalazimo sve uglove na slici, ukoliko uspesno pronadje 54 ugla sahovske table, ret ce biti TRUE, a u corners smestamo koordinate uglova na slici
    ret, corners = cv2.findChessboardCorners(gray, chessboard_corners, None) #ako zelimo poseban nacin detektovanja, to mozemo zameniti umesto None

    if ret:
        matrix_points.append(matrix)  # Dodajemo trodimenzijalne tacke sahovske table
        image_points.append(corners)  # Dodajemo koordinate uglova na slici



# Sada mozemo da izracunamo kalibracione parametre
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(matrix_points, image_points, gray.shape[::-1], None, None)
print("/**************** Camera Matrix ****************/")
print(camera_matrix)

print("\n/*************** Distortion Coefficients *****************/")
print(dist_coeffs)

# Sada mozemo da ispravimo iskrivljene slike
img = cv2.imread('camera_cal\calibration1.jpg')
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
cv2.imshow('Distorted image: ', img)
cv2.imshow('Undistorted image: ', undistorted_img)
cv2.waitKey(0) #slike ce ostati prikazane sve dok ih rucno ne uklonimo
cv2.destroyAllWindows()
