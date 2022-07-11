from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# load and prepare the image
def load_image_from_canvas(filename):
    image = load_img(filename, grayscale=True, target_size=(28, 28))
    image = img_to_array(image).astype('float32')
    image = image / 255.0
    return image

