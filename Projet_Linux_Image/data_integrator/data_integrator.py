# Importation des librairies:
import os
import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

# convertir le tenseur en image:
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)



# Define the functions for loading and displaying images here
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


# Then, your existing functions and main routine follow
def list_images_from_folder(folder_path):
    """Lists the images in the specified folder and returns them as a list."""
    images = [img for img in os.listdir(folder_path) if
              img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jfif'))]
    return images


def select_image(images_list, folder_name):
    """Displays the images list for user selection and returns the selected image path."""
    print(f"Available images in {folder_name}:")
    for i, image in enumerate(images_list, start=1):
        print(f"{i}. {image}")
    choice = input(f"Select an image number from {folder_name} (1-{len(images_list)}): ")
    if choice.isdigit() and 1 <= int(choice) <= len(images_list):
        return images_list[int(choice) - 1]
    else:
        print("Invalid selection. Please try again.")
        return select_image(images_list, folder_name)  # Recursive call to handle invalid input


def main():
    content_folder_path = './content_images'
    style_folder_path = './style_images'

    content_images = list_images_from_folder(content_folder_path)
    style_images = list_images_from_folder(style_folder_path)

    selected_content_image = select_image(content_images, './Projet_Linux_Image/content_images')
    selected_style_image = select_image(style_images, './Projet_Linux_Image/style_images')

    content_path = os.path.join(content_folder_path, selected_content_image)
    style_path = os.path.join(style_folder_path, selected_style_image)

    return content_path, style_path  # Return the paths


if __name__ == "__main__":
    content_path, style_path = main()  # Capture the returned paths

    # Load and display the images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    print("Lecture et affichage possible !")
