## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

	1.1 Pre svega spram dimenzija sahovske table sa nasih slika, potrebno je definisati matricu sa trodimenzionalnim koordinatama.
 
	1.2 Zatim svaku sliku moramo da pretvorimo u sive nijanse pomocu funkcije cvtColor() iz CV biblioteke, jer se detektovanje ivica vrsi spram nijanse svetline, a ne na osnovu boja.
 
	1.3 Koordinate ivica dobijamo funkcijom findChessboardCorners(), koja nam takodje vraca podatak da li je objekat odnosno nasa sahovska tabla dimenzija 10x7(odnosno uglova 9x6) pronadjena.
 
	1.4 Sada mozemo pomocu funkcije calibrateCamera() da izracunamo matricu kamere(fokalna duzina, opticki centar, skaliranje), kao i koeficijente distorzije.
 
	1.5 Prosledjivanjem ovih parametara funkciji undistort() dobijamo kao njenu povratnu vrednost ispravljenu sliku.
 
	1.6 Sliku mozemo prikazati pomocu funkcije imshow().
 
 	1.7 U prilogu mozemo videti kako izgleda originalna slika('Distored') i ista nakon primene undistort() funkcije('Undistored').
	
![Undistored](https://github.com/user-attachments/assets/e4a825fe-65e7-4380-894d-1a650d398e28) 
![Distored](https://github.com/user-attachments/assets/509db27e-00d7-46c4-ad75-122ad0102c36)






#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

	2.1 Odlucila sam se za kombinaciju Canny algoritma i LAB color space-a, kako bih uspesno detektovala i bele i zute linije na putu.
 
	2.2 Canny algoritmu prosledjujemo sliku u sivoj nijansi, jer cemo tako lakse detektovati promenu inteziteta piksela. On je sporiji, ali precizniji od Sobel algoritma. 
 
	2.3 Originalnu sliku konvertujemo i u LAB color space i izdvajamo B kanal koji predstavlja nijasnu zuto-plava i postavljamo thresholde prilagodjene 	nijansama zute boje.
 
	2.4 Na cisto crnoj slici isrtavamo bele piksele koji zadovoljavaju jedan od prethodna dva algoritma, cime dobijamo skup detekcije bele i zute linije na 	putu.
 
 	2.5 U prilogu mozemo videti originalnu sliku , kao i binarnu sliku sa odredjenim pragovima


![Binary_base](https://github.com/user-attachments/assets/e073391d-daf3-433b-bd48-b3b36e49732f)
![Binary_new](https://github.com/user-attachments/assets/a5d59173-1061-417d-9c3a-282edddd6097)

#### 3. Describe how (and identify where in your code) you performed a pers pective transform and provide an example of a transformed image.

        3.1 Za transformaciju perspektive najpre je bilo potrebno sa bazne slike odrediti deo koji zelimo da transformisemo, sto je odradjeno zadavanjem 4 tacke pravougaonika.
	
	3.2 Naspram baznog pravougaonika definisemo pravougaonik na destinacionoj slici, sa definisanim ofsetom koji menja sirinu destinacionog pravougaonika.
 
	3.3 Prosledjivanjem tih parametara funkciji cv2.getPerspectiveTransform(src, dst), dobijamo matricu transformacije.
 
	3.4 Zatim funkcija cv2.warpPerspective() na osnovu matrice transformacije, bazne slike i kljucnih tacaka kreira novu sliku sa pticijom perspektivom.
 
	3.5 U prilogu mozemo videti binarnu sliku u originalnoj perspektivi (sa redukovanim pragovima obrade spram slike postavljene u prethodnom delu zadatka - smanjeni sumovi), kao i istu iz pticije perspektive.
![binary_persp](https://github.com/user-attachments/assets/afe68e76-dab9-4a95-bb0e-0d7885c0d0da)
![Perspective_tran1](https://github.com/user-attachments/assets/06c4855d-360c-4785-9890-2133959f1140)




#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

	4.1 Koristila sam binarnu sliku nakon perspektivne transformacije (img_pt) kako bih pronašla sve bele piksele koji potencijalno pripadaju trakama.
 
 	4.2 Izracunavanje histograma: Izračunala sam histogram slike koristeći funkciju np.sum, kako bih analizirala intenzitet belih piksela duž horizontalne ose.
  
  	4.3 Histogram pomaže da identifikujem početne pozicije leve i desne trake, left_base i right_base, kao najviše vrednosti u odgovarajućim polovima histograma.
   
   	4.4 Implementirala sam metodu kliznog prozora. Podešeni su pravougaoni prozori širine margin (100) i visine window_height (80).
    
    	4.5 Prolazeći vertikalno duž slike, iterativno sam identifikovala bele piksele unutar prozora koristeći logičke maske, i to za levu (good_left_inds) i desnu traku (good_right_inds).
     
	4.6 Bele piksele koji pripadaju trakama sakupljam u listama left_lane_inds i right_lane_inds. Na kraju, kombinujem sve ove piksele koristeći np.concatenate.
 
	4.7 Koristeći funkciju np.polyfit, aproksimirala sam piksele leve i desne trake drugim stepenom polinoma. Ovo mi omogućava da definišem matematičku funkciju koja prati oblik svake trake.
 
	4.8. Generisala sam niz tačaka ploty, a zatim koristila koeficijente polinoma (left_fit i right_fit) za izračunavanje odgovarajućih x-koordinata traka (left_fitx i right_fitx).
 
 	4.9 Na kraju, nacrtala sam aproksimirane trake koristeći koordinate fitovanih polinoma, bojanjem odgovarajucih piksela.
![LaneDetectionFittedPoly](https://github.com/user-attachments/assets/17b588a4-fe27-4588-8cef-4fe11ebd459e)



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

	5.1 Izračunavanje radijusa zakrivljenosti: Izračunavanje koeficijenata u metrima (left_fit_cr, right_fit_cr): 
		left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

	5.2 Radijus zakrivljenosti se računa pomoću: R= (1+(2Ay+B)^2)^3/2 / (2A), gde su A,B i C koeficijenti parabole, a y predstavlja tacku na kojoj se radijus racuna. 
 
	5.3 Prosečan radijus: Kombinuje radijuse leve i desne trake: avg_curverad = (left_curverad + right_curverad) / 2. Sto je radijus veci, put je ravniji.
 
	5.4 Pozicija vozila se računa kao razlika između centra slike (koji predstavlja poziciju vozila) i centra trake. Ova vrednost se zatim konvertuje u metre koristeći faktor xm_per_pix. 
 
	5.5 Ako je center_offset pozitivan, vozilo je desno od centra trake , ako je negativan, vozilo je levo.
 
	5.6 Numerički rezultati (radijus zakrivljenosti i pozicija vozila) se ispisuju direktno na konačnu sliku koristeći OpenCV funkciju cv2.putText()

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

	6.1 Nakon detekcije traka i prilagođavanja polinoma, kreirana je maska koja iscrtava područje između leve i desne trake, simulirajući njihov položaj na slici.
 
	warp_zero = np.zeros_like(img_pt).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
 
	6.2 Ovime se između parabola iscrtava zatvoreni zeleni pravougaonik.
 
	6.3 Zatim je odradjena obrnuta perspektivna transformacija dst-> src, kako bi se vratila prvobitna perspektiva
 
	6.4 Zatim je ova slika kombinovana sa originalnom slikom sa ispravljenom distorzijom, kako bi dobili jasnu sliku konacne obrade:
	result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
	
![Result](https://github.com/user-attachments/assets/b85079ce-133c-45d2-b421-1d51afc7d7e8)

	
### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?



