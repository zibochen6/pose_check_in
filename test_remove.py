from rembg import remove
import cv2

input_path = '/home/seeed/demo/punch_in/punch_photos/punch_20251027_094026_anime.jpg'
output_path = './output.png'

input = cv2.imread(input_path)
output = remove(input)
cv2.imwrite(output_path, output)