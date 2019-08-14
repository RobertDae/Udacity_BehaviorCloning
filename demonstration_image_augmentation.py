import cv2
import utils 

img_left_original = cv2.imread("./MyData06/IMG/left_2019_08_11_15_35_48_328.jpg")
img_center_original = cv2.imread("./MyData06/IMG/center_2019_08_11_15_35_48_328.jpg")
img_right_original = cv2.imread("./MyData06/IMG/right_2019_08_11_15_35_48_328.jpg")

img_l_o_rgb = utils.bgr2rgb(img_left_original)
img_c_o_rgb = utils.bgr2rgb(img_center_original)
img_r_o_rgb = utils.bgr2rgb(img_right_original)

img_left_cropped = utils.crop_and_resize(img_l_o_rgb)
img_center_cropped = utils.crop_and_resize(img_c_o_rgb)
img_right_cropped = utils.crop_and_resize(img_r_o_rgb)
img_left_cropped = cv2.cvtColor(img_left_cropped, cv2.COLOR_RGB2BGR)
img_center_cropped =  cv2.cvtColor(img_center_cropped, cv2.COLOR_RGB2BGR)
img_right_cropped =  cv2.cvtColor(img_right_cropped, cv2.COLOR_RGB2BGR)
cv2.imwrite("./images/img_left_cropped.jpg", img_left_cropped )
cv2.imwrite("./images/img_center_cropped.jpg", img_center_cropped)
cv2.imwrite("./images/img_right_cropped.jpg", img_right_cropped)

img_left_flipped = utils.flipimg(img_left_cropped)
img_center_flipped = utils.flipimg(img_center_cropped)
img_right_flipped =  utils.flipimg(img_right_cropped)
cv2.imwrite("./images/img_left_flipped.jpg", img_left_flipped )
cv2.imwrite("./images/img_center_flipped.jpg", img_center_flipped)
cv2.imwrite("./images/img_right_flipped.jpg", img_right_flipped)
