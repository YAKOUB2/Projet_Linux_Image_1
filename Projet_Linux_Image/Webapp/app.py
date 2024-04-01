import os
import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import time

# Convertir le tenseur en image PIL
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Définir les fonctions de chargement et d'affichage des images
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

# Définir la fonction pour extraire les couches VGG
def vgg_layers(layer_names, style_path):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ["Introduction", "Select Images"])

    if page == "Introduction":
        st.title("Transformez vos photos en chefs-d'œuvre artistiques avec Artify !")
        st.write("Artify est une application novatrice de transfert de style neuronal, qui vous permet de transformer vos photos en véritables œuvres d'art en appliquant différents styles artistiques. Avec Artify, explorez un monde où la technologie rencontre l'art, et donnez vie à vos souvenirs en leur conférant des esthétiques variées telles que l'abstrait, l'impressionnisme ou encore le cubisme. Laissez libre cours à votre créativité et découvrez de nouvelles perspectives visuelles avec Artify dès aujourd'hui !")

        # Ajouter l'image à l'introduction
        intro_image_url = "https://miro.medium.com/v2/resize:fit:1400/1*8bbp3loQjkLXaIm_QBfD8w.jpeg"
        st.image(intro_image_url, caption="Introduction Image", use_column_width=True)

        st.header("Donnez vos images en entrées")
        # Charger et afficher les deux premières images côte à côte
        col1, col2 = st.columns(2)
        with col1:
            image_path_1 = "./exemple/Elia_Rahari.jpeg"  # Chemin de votre première image
            image1 = PIL.Image.open(image_path_1)
            st.image(image1, caption="Image 1", use_column_width=True)

        with col2:
            image_path_2 = "./exemple/Starry_night.jpg"  # Chemin de votre deuxième image
            image2 = PIL.Image.open(image_path_2)
            st.image(image2, caption="Image 2", use_column_width=True)


        st.header("Générez votre image")
        # Charger et afficher la troisième image juste en dessous des deux premières
        image_path_3 = "exemple/result_image.png"  # Chemin de votre troisième image
        image3 = PIL.Image.open(image_path_3)
        st.image(image3, caption="Image 3", use_column_width=True)

    elif page == "Select Images":
        st.title("Artify - Select Images")
        st.write("Select images from the available options:")

        content_folder_path = './content_images'
        style_folder_path = './style_images'

        content_images = os.listdir(content_folder_path)
        style_images = os.listdir(style_folder_path)

        selected_content_image = st.sidebar.selectbox("Select a content image:", content_images)
        selected_style_image = st.sidebar.selectbox("Select a style image:", style_images)

        quality_options = {"Faible": 3, "Moyen": 7, "Élevé": 10}
        selected_quality = st.sidebar.selectbox("Niveau de qualité:", options=list(quality_options.keys()))

        epochs = quality_options[selected_quality]

        content_path = os.path.join(content_folder_path, selected_content_image)
        style_path = os.path.join(style_folder_path, selected_style_image)

        content_image = load_img(content_path)
        style_image = load_img(style_path)

        # Afficher les images sélectionnées dans un conteneur
        st.subheader("Selected Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(tensor_to_image(content_image), caption="Selected Content Image", use_column_width=True)
        with col2:
            st.image(tensor_to_image(style_image), caption="Selected Style Image", use_column_width=True)

        if st.button('Run Model'):
            st.info("Running the model...")

            # Les couches intérmédiaires:
            content_layers = ['block5_conv2']

            style_layers = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1',
                            'block4_conv1',
                            'block5_conv1']

            # Extraire les caractéristiques de style:
            style_extractor = vgg_layers(style_layers, style_path)
            style_outputs = style_extractor(style_image*255)

            def gram_matrix(input_tensor):
                result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
                input_shape = tf.shape(input_tensor)
                num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
                return result / (num_locations)

            # Définir la classe de modèle StyleContentModel
            class StyleContentModel(tf.keras.models.Model):
                def __init__(self, style_layers, content_layers):
                    super(StyleContentModel, self).__init__()
                    self.vgg = vgg_layers(style_layers + content_layers, style_path)
                    self.style_layers = style_layers
                    self.content_layers = content_layers
                    self.num_style_layers = len(style_layers)
                    self.vgg.trainable = False

                def call(self, inputs):
                    inputs = inputs * 255.0
                    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
                    outputs = self.vgg(preprocessed_input)
                    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                                      outputs[self.num_style_layers:])

                    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

                    content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
                    style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

                    return {'content': content_dict, 'style': style_dict}

            extractor = StyleContentModel(style_layers, content_layers)
            style_targets = extractor(style_image)['style']
            content_targets = extractor(content_image)['content']

            # Définir la fonction de perte de style et de contenu
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

            def clip_0_1(image):
                return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

            @tf.function()
            def train_step(image):
                with tf.GradientTape() as tape:
                    outputs = extractor(image)
                    loss = style_content_loss(outputs)
                    loss += 30 * tf.image.total_variation(image)

                grad = tape.gradient(loss, image)
                opt.apply_gradients([(grad, image)])
                image.assign(clip_0_1(image))

            # Convertir l'image de contenu en variable tensor
            image = tf.Variable(content_image)

            # Définir l'optimisateur
            opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

            # Entrainement du modèle
            start = time.time()

            steps_per_epoch = 100

            step = 0
            progress_bar = st.progress(0)
            for n in range(epochs):
                for m in range(steps_per_epoch):
                    step += 1
                    train_step(image)
                progress_bar.progress((n + 1) / epochs)

            progress_bar.empty()

            # Enregistrer l'image résultante sur le disque
            result_image = tensor_to_image(image)
            st.image(result_image, caption="Resulting Image")

if __name__ == "__main__":
    main()
