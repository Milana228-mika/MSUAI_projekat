import cv2
import numpy as np
import glob


def main():
#***************************Compute the camera calibration matrix and distortion coefficients given a set of chessboard images************************

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


    #*************************** Apply a distortion correction to raw images ************************************

    # Sada mozemo da ispravimo iskrivljene slike
    img = cv2.imread('test_images\whiteCarLaneSwitch.jpg')
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    #cv2.imshow('Distorted image: ', img)
    #cv2.imshow('Undistorted image: ', undistorted_img)
    #cv2.waitKey(0) #slike ce ostati prikazane sve dok ih rucno ne uklonimo
    #cv2.destroyAllWindows()

    #************************** Use color transforms, gradients, etc., to create a thresholded binary image ************************************

    # Koristicemo Canny algoritam za detekciju linija na putu
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)  # To cemo mnogo lakse uraditi sa slikom u nijansama sive boje, jer se tako lakse detektuje promena inteziteta piksela
    low_threshold = 100  # Donji prag za Canny (sto je nizi detektujemo vise ivica ali se dobija i vise suma)
    high_threshold = 250  # Gornji prag za Canny ( sto je visi, manje ivica ali i manje suma)
    canny_edges = cv2.Canny(gray, low_threshold, high_threshold) #primena Canny algoritma

    # LAB color space i pragiranje B kanala - ovo radimo kako bismo detektovali i zute linije na putu
    lab = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2LAB) # konvertovanje u LAB color space
    b_channel = lab[:, :, 2]  # Samo B kanal! (zato sto on predstavlja nijansu zuta-plava)
    b_thresh_min = 150
    b_thresh_max = 255 #ovo su vrednosti prilagodjenje za zutu boju
    binary_b = np.zeros_like(b_channel) #najpre pravimo crnu sliku velicine kao b_channel
    binary_b[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1 #sada primenjujemo nase tresholde na B kanal 

    # Kombinujemo rezultate kako bismo dobili precizniju detekciju, i za zute i za bele linije
    combined_binary = np.zeros_like(binary_b)
    combined_binary[(binary_b == 1) | (canny_edges > 0)] = 1  
    binary_img = combined_binary * 255
    # Prikaz rezultata
    #cv2.imshow('Base image', image)
    cv2.imshow('Combined Binary', combined_binary * 255) #mnozimo sa 255 jer ce crni deo (0) i dalje ostati 0, a sivi deo(1) ce postati 255 odnosno beo za imshow
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #*************************** Apply a perspective transform to rectify binary image ("birds-eye view") ************************************

    line_dst_offset = 200  # Veći offset - sira perspektiva
    src = [  #koordinate 4 tacke sa ulazne slike 
        [450, 380],  #gornje tacke
        [830, 380],  #gornje tacke
        [1200, binary_img.shape[0]], #donje tacke 
        [100, binary_img.shape[0]]   #donje tacke
    ]

    dst = [  # tacke na novoj slici, pomeramo ih tako da prethodno odabrani pravougaonik sada pretstavlja citavu sliku
        [src[3][0] + line_dst_offset, 0],                
        [src[2][0] - line_dst_offset, 0],                  
        [src[2][0] - line_dst_offset, binary_img.shape[0]], 
        [src[3][0] + line_dst_offset, binary_img.shape[0]]  
    ]

    img_pt = PerspectiveTransform(binary_img, src, dst)
    cv2.imshow('PerspectiveTransform image', img_pt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def PerspectiveTransform(img, src, dst):
    src = np.float32(src)  
    dst = np.float32(dst)

    # Primena perspektivne transformacije
    transform_matrix = cv2.getPerspectiveTransform(src, dst) #matrica transofrmacije perspektive
    return cv2.warpPerspective(img, transform_matrix, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


                               
if __name__ == '__main__':
    main()