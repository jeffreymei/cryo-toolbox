import cv2 
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import rasterio
import sys

if len(sys.argv) > 1:
	filename = sys.argv[1]
else:
	ls = sorted(glob.glob("/home/jeffrey/Dropbox (MIT)/Images/Lincoln Sea/S2A_MSIL1C_20180731T201851_N0206_R071_T20XNS_20180731T234334/*.jp2"))
	filename = ls[0]

with rasterio.open(filename) as f:
	im_bw = f.read(1)
im_bw = (im_bw/float(im_bw.max())*255).astype('uint8')
im_c = cv2.cvtColor(im_bw,cv2.COLOR_GRAY2RGB)


#tolerance for floes that touch image edges
#this is important to exclude floes that are only partially in the image
tol = int(np.sqrt(im_bw.size)/100)

#this function does nothing but is required as an input for cv2 functions
def nothing(x):
	pass

print("Press <space> when you have happy with the floe delineations. \n Press <ESC> if you want to quit without saving floe areas.")

#Below section finds best rectangular contour
#then rotates it to make the longest edge flat
#this is technically not required but if you have
#images at weird angles, it improves viewing size
#if you want to ignore this section, just set
#dst = im_bw in line 54
ret, img = cv2.threshold(im_bw, 0,0, cv2.THRESH_TOZERO)
contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#below assertion checks that the image boundary contour 
#has been identified. if it fails, then the image boundary
#was not identified
try: assert(len(contours)==1)
except: raise(Exception)
cnt = contours[0]
rect = cv2.minAreaRect(cnt)
angle = rect[-1]
M = cv2.getRotationMatrix2D((im_bw.shape[1]/2,im_bw.shape[0]/2),90+angle, 1)
dst = cv2.warpAffine(im_bw,M,(im_bw.T.shape))
#need to make dst 3-channel so that colored lines can be drawn
dst_c = cv2.cvtColor(dst,cv2.COLOR_GRAY2RGB)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#kernel affects the size of each opening/close iteration
cv2.createTrackbar('kernel', 'image', 1,5, nothing)
#number of open iterations
cv2.createTrackbar('open_iter', 'image', 2, 11, nothing)
#number of close iterations
cv2.createTrackbar('close_iter', 'image', 0, 11, nothing)
#image 'color' threshold, areas darker than this are deleted as 'water'
cv2.createTrackbar('cutoff', 'image', 220, 255, nothing)
#thickness of the red lines that delineate contours
cv2.createTrackbar('line thickness', 'image', 6, 20, nothing)
cv2.resizeWindow('image', (int(im_bw.shape[1]/im_bw.shape[0]*600), 600))
while (1):
	K_size = cv2.getTrackbarPos('kernel', 'image')*2+1
	n_open = cv2.getTrackbarPos('open_iter', 'image')
	n_close = cv2.getTrackbarPos('close_iter', 'image')
	thick = cv2.getTrackbarPos('line thickness', 'image')
	cutoff = cv2.getTrackbarPos('cutoff', 'image')
	k = cv2.waitKey(1) & 0xFF

	#threshold rotated image
	ret, filt = cv2.threshold(dst, cutoff,0, cv2.THRESH_TOZERO)
	
	#get image boundary i.e. rotated rectangle 
	ret, img = cv2.threshold(dst, 0,0, cv2.THRESH_TOZERO)
	contours1, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	assert(len(contours1)==1)
	cnt = contours1[0]
	rot_rect = cv2.minAreaRect(cnt)
	filt = cv2.medianBlur(filt , 5) #you can try different blurs...
	# filt = cv2.bilateralFilter(filt,7,11,11)
	kernel = np.ones((K_size, K_size),np.uint8)
	closing = cv2.morphologyEx(filt, cv2.MORPH_OPEN, kernel, iterations=n_open)
	filt2 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=n_close)
	filt2_c = cv2.cvtColor(filt2,cv2.COLOR_GRAY2RGB)
	contours, hierarchy = cv2.findContours(filt2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	
	#calculate areas of floes (in pixels)
	areas = np.zeros(len(contours))
	cxs = np.zeros(len(contours), dtype='int')
	cys = np.zeros(len(contours), dtype='int')
	for i in range(len(contours)):
		cnt = contours[i]
		M = cv2.moments(cnt)
		if M['m00']!=0:
			cxs[i] = int(M['m10']/M['m00'])
			cys[i] = int(M['m01']/M['m00'])
			areas[i] = cv2.contourArea(cnt)
		else:
			print("Zero moment; check thresholds too low")

	desc = np.argsort(-areas)
	minx, maxx = rot_rect[0][0]-rot_rect[1][0]/2, rot_rect[0][0]+rot_rect[1][0]/2
	miny, maxy = rot_rect[0][1]-rot_rect[1][1]/2,rot_rect[0][1]+rot_rect[1][1]/2
	
	#show contours in image
	for i in range(len(desc[desc>0])):
		j = desc[i]
		contoursx, contoursy = contours[j].T[0][0],contours[j].T[1][0]
		if np.sum(contoursx>maxx-tol)+np.sum(contoursx<minx+tol)+np.sum(contoursy>maxy-tol)+np.sum(contoursy<miny+tol)!=0:
			# print("touched boundary")
			cv2.drawContours(filt2_c, contours[j], -1, (192,192,192), thick)
			# cv2.putText(output,  "{:.0f}".format(areas[j]/1e3), (cxs[j], cys[j]), cv2.FONT_HERSHEY_SIMPLEX, factor, (192,192,192), thickness=thick)
			areas[i] = 0 #do not save these

		else:
			cv2.drawContours(filt2_c, contours[j], -1, (0,0,255), thick)
			# cv2.putText(filt2_c,  "{:.0f}".format(areas[j]), (cxs[j], cys[j]), cv2.FONT_HERSHEY_SIMPLEX, factor, (0,0,255), thickness=thick)
	
	if k == 27: #esc to exit without saving
		break
	elif k == 32: #space to save floe areas and move on to next img
		print("{} floes found".format(len(areas[areas>0])))
		files = np.array([filename[:-4]]*len(areas[areas>0]))
		save_array = np.vstack((files, areas[areas>0])).T
		with open("floe_areas.txt", "ab") as f:
			np.savetxt(f, save_array, fmt='%s %s')
		print("Now hit ESC to close this window")
	output = np.hstack((dst_c, filt2_c)) 
	cv2.imshow("image", output)
cv2.destroyAllWindows()

# plt.subplot(211)
# #Raw image
# plt.imshow(im_bw, cmap='Greys_r')
# plt.suptitle("FSD calculator using grayscale image")

# # plt.subplot(312)
# ##Rotate to be level; also, floes that touch this boundary will be discarded
# # plt.title("Rotated")

# plt.subplot(212)
# # identify contours; sort by area; discard small floes + floes that touch box
# plt.title("Floes with areas, excluding protruding floes (gray)")

# plt.imshow(filt2, cmap='Greys_r')

# # plt.plot([minx, maxx], [miny, maxy])
# plt.show()

