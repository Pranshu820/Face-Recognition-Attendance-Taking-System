# HOG or Histogram of Oriented Gradients is a feature descriptor used in Computer Vision.
# This technique counts occurrences of gradient orientation in the localized portion of an image.

# Procedure:
# 1) The image is first converted into BGR image(Blue, Green, Red) to simplify the procedure.
# 2) Then, an arrow (->) is drawn from where the shade is light to where the shade is dark.
# 3) These arrows are known as gradients.
# 4) Finally, after drawing all the arrows, the picture obtained is the HOG of that image.

# Importing all important dependencies
from skimage.feature import hog
from skimage import exposure
from matplotlib import pyplot as plt
import cv2

# imread function of cv2 loads an image from a specified file
# You can put the path of the image you want to create the HOG for in the imread function
image = cv2.imread('/Users/pranshu/Desktop/Face_Rec/Images/S.Ramos.jpeg')

# Defining the dimensions and other important values to the HOG
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')

# Converting the original image to BGR image
ax1.imshow(image, cmap=plt.cm.gray)

ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)

ax2.set_title('Histogram of Oriented Gradients(HOG)')

# Printing the original and HOG image
plt.show()
