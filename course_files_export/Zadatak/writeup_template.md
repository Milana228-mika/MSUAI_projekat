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

	1.1 Pre svega spram dimenzija sahovske table sa nasih slika, potrebno je definisati matricu sa trodimenzionalnim koordinatama
	1.2 Zatim svaku sliku moramo da pretvorimo u sive nijanse pomocu funkcije cvtColor() iz CV biblioteke, jer se detektovanje ivica vrsi spram nijanse svetline, a ne spram boja
	1.3 Koordinate ivica dobijamo funkcijom findChessboardCorners(), koja nam takodje vraca podatak da li je objekat odnosno nasa sahovska tabla dimenzija 10x7(odnosno uglova 9x6) pronadjena
	1.4 Sada mozemo pomocu funkcije calibrateCamera() da izracunamo matricu kamere(fokalna duzina, opticki centar, skaliranje), kao i koeficijente distorzije
	1.5 Prosledjivanjem ovih parametara funkciji undistort() dobijamo kao njenu povratnu vrednost ispravljenu sliku
	1.6 Sliku mozemo prikazati pomocu funkcije imshow()
	
![Undistored](https://github.com/user-attachments/assets/e4a825fe-65e7-4380-894d-1a650d398e28) 
![Distored](https://github.com/user-attachments/assets/509db27e-00d7-46c4-ad75-122ad0102c36)






#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

	2.1 Odlucila sam se za kombinaciju Canny algoritma i LAB color space-a, kako bih uspesno detektovala i bele i zute linije na putu
	2.2 Canny algoritmu prosledjujemo sliku u sivoj nijansi, jer cemo tako lakse detektovati promenu inteziteta piksela
	2.3 Zatim originalnu sliku konvertujemo i u LAB color space i izdvajamo B kanal koji predstavlja nijasnu zuto-plava i postavljamo thresholde prilagodjene 	nijansama zute boje
	2.4 Zatim na cisto crnoj slici isrtavamo bele piksele koji zadovoljavaju jedan od prethodna dva algoritma, cime dobijamo detekciju i bele i zute linije na 	putu


![Binary_base](https://github.com/user-attachments/assets/e073391d-daf3-433b-bd48-b3b36e49732f)
![Binary_new](https://github.com/user-attachments/assets/a5d59173-1061-417d-9c3a-282edddd6097)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

