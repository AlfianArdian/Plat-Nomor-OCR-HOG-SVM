# menggunakan program dan foldernya
# python ocr_license_plate.py --input license_plates/group1
# python ocr_license_plate.py --input license_plates/group2 --clear-border 1

#import packages yang dibutuhkan termasuk packages
#custom PlateRecognition_Engine
from ocrmobil.anpr import PlateRecognition_Engine
from imutils import paths
import argparse
import imutils
import cv2

def cleanup_text(text):
	#pembersihan string untuk membantu fungsi dari openCV 'cv2.putText'
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# membuat argumen parser dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
	help="whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=7,
	help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not to show additional visualizations")
args = vars(ap.parse_args())
#import telah dilakukan, utilitas pembersihan teks dilakukan, dan
#pendefinisian argumen command line

# menginisialisasi engine plat nomor
anpr = PlateRecognition_Engine(debug=args["debug"] > 0)

# ambil semua gambar yang diinputkan
imagePaths = sorted(list(paths.list_images(args["input"])))

# looping semua gambar yang diinputkan
for imagePath in imagePaths:
	# load gambar input lalu resize
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)

	#terapkan pendeteksi plat nomor dari lpText, dan lpCnt
	#yang didefinisikan
	(lpText, lpCnt) = anpr.find_and_ocr(image, psm=args["psm"],
		clearBorder=args["clear_border"] > 0)

	# hanya lanjutkan program jika gambar plat nomor berhasil di OCR
	if lpText is not None and lpCnt is not None:
		# kalkulasikan dan gambar bounding box dari kontur plat nomor
		box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
		box = box.astype("int")
		cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

		# beri anotasi pada 'lpText' yang telah dibersihkan
		(x, y, w, h) = cv2.boundingRect(lpCnt)
		cv2.putText(image, cleanup_text(lpText), (x, y - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

		# tampilkan output dari gambar yang diproses ANPR
		print("[INFO] {}".format(lpText))
		cv2.imshow("Output ANPR", image)
		cv2.waitKey(0)