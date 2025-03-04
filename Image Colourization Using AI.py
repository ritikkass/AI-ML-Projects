## 5. Image Colorization Using AI

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
gray = cv2.imread('grayscale.jpg', 0)
colorized = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

plt.imshow(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
plt.show()
