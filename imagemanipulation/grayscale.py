# Neha Srisatya Pithani
# Fall Semester, 2017
# this little algorithm converts an image, in this case chill.jpg, into a grayscale image
# this demonstrates use of numpy and matplotlib libraries

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


fname='chill.jpg'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
plt.imshow(arr, cmap='gray')
plt.show()