# import package yang diperlukan
from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

class PlateRecognition_Engine:
	def __init__(self, minAR=4, maxAR=5, debug=False):
		# aspek rasio minimum dan maksimum untuk membatasi kotak plat nomor
		# agar tiap proses ditampilkan resultnya
		self.minAR = minAR
		self.maxAR = maxAR
		self.debug = debug

	def debug_imshow(self, title, image, waitKey=False):
		# cek apakah kita di status debug jika iya maka
		# tampilkan gambar
		if self.debug:
			cv2.imshow(title, image)

			# menunggu tombol yang ditekan di keyboard
			if waitKey:
				cv2.waitKey(0)

	def locate_license_plate_candidates(self, gray, keep=5):
		# menggunakan blackhat morphological operation untuk menampilkan
		# karakter yang gelap terhadap background yang terang dengan dimensi
		# lebar 13 dan tinggi 5, sesuai dengan tinggi plat nomor biasanya.
		rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
		self.debug_imshow("Blackhat", blackhat)

		# deteksi daerah gambar yang banyak light
		squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
		light = cv2.threshold(light, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		self.debug_imshow("Light Regions", light)

		# menggunakan cv2.sobel untuk komputasi magnitude gradien scharr
		# represntasi di x-direction dari gambar blackhat dan lalu scale
		# ulang untuk menghasilkan insensitas yang kembali ke range
		# [0,255]
		gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
			dx=1, dy=0, ksize=-1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
		gradX = gradX.astype("uint8")
		self.debug_imshow("Scharr", gradX)

		# mengaplikasikan gaussian blur untuk gambar magnitude gradien
		# lalu mengaplikasikan threshold binary dengan metode otsu
		gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
		thresh = cv2.threshold(gradX, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		self.debug_imshow("Grad Thresh", thresh)

		# melakukan erosi dan dilasi untuk membersihkan gambar yang
		# di treshold
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		self.debug_imshow("Grad Erode/Dilate", thresh)

		# syntax 'light' sebelumnya bertindak sebagai masking untuk
		# bitwise dan antara hasil treshold dan meampilkan bagian
		# gambar kandidat plat nomor, lalu dilakukan dilasi dan erosi
		# lagi
		thresh = cv2.bitwise_and(thresh, thresh, mask=light)
		thresh = cv2.dilate(thresh, None, iterations=2)
		thresh = cv2.erode(thresh, None, iterations=1)
		self.debug_imshow("Final", thresh, waitKey=True)

		# menutup bagian program yang menentukan lokasi kandidat plat
		# nomor dengan menemukan semua contours, kembali mengatur pixel
		#sambil mempertahankan contours
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

		#kembalikan hasil sortir dan pemangkasan
		return cnts

	def locate_license_plate(self, gray, candidates,
		clearBorder=False):
		#sebelum dimulai looping kandidat plat nomor, pertama initialize
        # variable yang dapat 'mengingat' contour (lpCnt) dan plat nomor region of interest (roi)
		lpCnt = None
		roi = None

		# loop pada kandidat contours plat nomor untuk mengisolasi contour
		# yang tedapat plat nomor dan ekstrak ROI dari plat nomornya itu sendiri
		for c in candidates:
			# selanjutnya menentukan kotak pembatas untuk contour (c)
			# menentukan aspek ratio contour dari kotak pembatas
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)

			# cek apakah aspek rasio sebuah kotak yang bisa diterima
			if ar >= self.minAR and ar <= self.maxAR:
				# set dari contours yang ada, ekstrak plat nomor dari gambar
				# grayscale dan biner-inverse threshold menggunakan metode otsu
				lpCnt = c
				licensePlate = gray[y:y + h, x:x + w]
				roi = cv2.threshold(licensePlate, 0, 255,
					cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
				#bagian locate_license_plate sudah selesai, lalu ke fase selanjutnya 
				# eliminasi noises jika ada di dekat plat nomor
				if clearBorder:
					roi = clear_border(roi)

				# menampilkan informasi debugging lalu break dari loop
				# karena telah menemukan region plat nomornya
				self.debug_imshow("License Plate", licensePlate)
				self.debug_imshow("ROI", roi, waitKey=True)
				break

		# mengembalikan 2-tuple roi plat nomor dan contours ke caller
		return (roi, lpCnt)
	#Proses morfological oleh opencv selesai dengan hasil gambar bersih
	#yang akan di proses oleh engine tesseract-ocr
	#page segmentation method (psm) dengan nilai 7
	def build_tesseract_options(self, psm=7):
		# list karakter yang tesseract pertimbangkan sebagai hasil OCR
		# lalu menggabungkan kedua string dengan parameter ini
		alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		options = "-c tessedit_char_whitelist={}".format(alphanumeric)

		# atur mode psm yang telah digabungkan
		options += " --psm {}".format(psm)

		# string options akan dikembalikan ke caller
		return options

	def find_and_ocr(self, image, psm=7, clearBorder=False):
		#bawa semua komponen jadi satu lalu diinisialisasi
		lpText = None

		# konversi gambar inputan ke grayscale
		# tentukan 'candidates' set plat nomor dari gambar 'gray' lewat
		# metode yang kita tentukan sebelumnya
		# temukan plat nomor dari 'candidates' yang menghasilkan 'lp' ROI
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		candidates = self.locate_license_plate_candidates(gray)
		(lp, lpCnt) = self.locate_license_plate(gray, candidates,
			clearBorder=clearBorder)

		# asumsi bahwa plat nomor yang dipilih sudah cocok
		if lp is not None:
			# lakukan ocr pada plat nomor lewat 'image_to_string'
			options = self.build_tesseract_options(psm=psm)
			lpText = pytesseract.image_to_string(lp, config=options)
			self.debug_imshow("License Plate", lp)

		# kembalikan 2 tuple yang terdiri dari OCR 'lpText' dan 'lpCnt'
		return (lpText, lpCnt)