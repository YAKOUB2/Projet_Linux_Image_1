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
    content_folder_path = '../content_images'
    style_folder_path = '../style_images'

    content_images = list_images_from_folder(content_folder_path)
    style_images = list_images_from_folder(style_folder_path)

    selected_content_image = select_image(content_images, 'content_images')
    selected_style_image = select_image(style_images, 'style_images')

    content_path = os.path.join(content_folder_path, selected_content_image)
    style_path = os.path.join(style_folder_path, selected_style_image)

    return content_path, style_path  # Return the paths


if __name__ == "__main__":
    content_path, style_path = main()  # Capture the returned paths

    # Load and display the images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

# Avoir les outputs des couches intermédiaires:
def vgg_layers(layer_names):
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model

# Les couches intérmédiaires:
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


# Extraire les caractéristiques de style:
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

# Capture des motifs et textures:
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

# Séparer les caractéristiques de style et de contenu:
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

# Application de la fonction:
extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Définir l'étape d'entrainement: fonction de loss
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs):
  style_outputs = outputs['style']
  content_outputs = outputs['content']
  style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                         for name in style_outputs.keys()])
  style_loss *= 1e-2 / len(style_layers)

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                           for name in content_outputs.keys()])
  content_loss *= 1e4 / len(content_layers)
  loss = style_loss + content_loss
  return loss

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
      outputs = extractor(image)
      loss = style_content_loss(outputs)
      loss += 30 * tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

# Convertir l'image de contenu en variable tensor:
image = tf.Variable(content_image)

# Définir l'optimisateur:
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


# Entrainement du modèle:
import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))

# Enregistrer l'image résultante sur le disque
result_image = tensor_to_image(image)
result_image.save("../exemple/result_image.png")

end = time.time()
print("Total time: {:.1f}".format(end-start))

