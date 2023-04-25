from infer import InferenceHelper
from PIL import Image
import matplotlib.pyplot as plt
import time


infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a single pillow image
img = Image.open("test_imgs/my room 002.jpg")  # any rgb pillow image
img = img.resize(size=(640, 480))
bin_centers, predicted_depth = infer_helper.predict_pil(img)

plt.imshow(predicted_depth[0][0], cmap='plasma')
plt.show()
