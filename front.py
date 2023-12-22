import streamlit as st
import cv2
import numpy as np

from PIL import Image

from super_image import EdsrModel, ImageLoader, MsrnModel
# import super_image
# import requests

def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
    image_orig = None
    image = None

    st.title("Обработка изображения")

    menu = ["Изображение","Видео"]
    choice = st.sidebar.selectbox("",menu)

    if choice == "Изображение":
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

        if image_file is not None:

                file_details = {"filename":image_file.name, "filetype":image_file.type,
                                "filesize":image_file.size}
                st.write(file_details)
                bytes_data = image_file.read()
                image_orig = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

                image = image_orig.copy()

                st.image(image, width=300)
    
    st.sidebar.checkbox("Удаление теней")

    st.sidebar.checkbox("Удаление бликов")

    st.sidebar.checkbox("Удаление размытия")

    st.sidebar.checkbox("Убрать естественные возмущения")

        # st.image(image_noise, width=300)

    # with st.sidebar.expander(label = "Test"):

    #     value_brightness = st.sidebar.slider("Яркость", 1, 100, value=1, step=1)

    #     value_contrast = st.sidebar.slider("Контраст", 1, 10, value=1, step=1)

    #     value_blur = st.sidebar.slider("Размытие", 1, 10, value=1, step=1)

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.number_input("test input")
        # with col2:
        #     st.number_input("this is my other test input")
    
    if st.sidebar.checkbox("Улучшение качества (Тяжелая модель)"):

        # model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)  
        model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=4)  

        image_np = Image.fromarray(np.uint8(image))

        inputs = ImageLoader.load_image(image_np)
        predict_image = model(inputs)
        ImageLoader.save_image(predict_image, './images/image.png')

        image = cv2.imread('./images/image.png')[:,:,::-1]

    if st.sidebar.checkbox("Улучшение качества (Средняя модель)"):

        # model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)  
        # model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=4)  
        model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2)

        image_np = Image.fromarray(np.uint8(image))

        inputs = ImageLoader.load_image(image_np)
        predict_image = model(inputs)
        ImageLoader.save_image(predict_image, './images/image.png')

        image = cv2.imread('./images/image.png')[:,:,::-1]

    if st.sidebar.checkbox("Улучшение качества (Легкая модель)"):

        # model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)  
        # model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=4)  
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)  

        image_np = Image.fromarray(np.uint8(image))

        inputs = ImageLoader.load_image(image_np)
        predict_image = model(inputs)
        ImageLoader.save_image(predict_image, './images/image.png')

        image = cv2.imread('./images/image.png')[:,:,::-1]

    if st.sidebar.checkbox("Убрать шум"):
    
        image = cv2.fastNlMeansDenoisingColored(image, None,10,10,7,21)

    sidebar_expander = st.sidebar.expander("Ручное регулирование")
    with sidebar_expander:

        value_brightness = st.slider("Яркость", 1, 100, value=1, step=1)
        value_contrast = st.slider("Контраст", 1, 10, value=1, step=1)
        value_blur = st.slider("Размытие", 1, 10, value=1, step=1)




    if image is not None:

        image = cv2.convertScaleAbs(image, alpha=value_contrast, beta=value_brightness)

        # lookup_table = np.array([((i / 255.0) ** value_brightness) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # gamma_corrected_image = cv2.LUT(image_orig, lookup_table)

        image = cv2.blur(image, (value_blur, value_blur))

        st.image(image, width=300)

    if st.sidebar.button("Вернуть исходное изображение"):
          image = image_orig.copy()
    


if __name__ == '__main__':
	main()