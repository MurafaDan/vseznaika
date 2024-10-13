from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('final_model.keras')

img = image.load_img('102841525_bd6628ae3c.jpg', target_size=(180, 180))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
print(f"The image belongs to class: {class_names[predicted_class[0]]}")
