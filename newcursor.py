import cv2

img = cv2.imread('E:\\GD\\facelandmarks_wingloss-master\\test (2).jpeg')
center = (100, 100)
radius = 50
color = (0, 0, 255)
thickness = 2
print(img.shape)
print(type(img))
cv2.circle(img, center, radius, color, thickness)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
