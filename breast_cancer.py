# Imports
import time
import streamlit as st
import numpy as np
from skimage import io
import cv2

# Model related imports
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms
import torch
import pickle
from question_generation.pipelines import pipeline

# Image transformations for DenseNet model
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Load main models
@st.cache(allow_output_mutation=True,
          hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()})
def load_models(root):
    # Load DenseNet-121 model
    with open(root + 'breast_tumor_model.pkl', 'rb') as f:
        model = pickle.load(f)

    model.eval()

    # Load gradcam for XAI
    with open(root + 'gradcam.pkl', 'rb') as f:
        gradcamplusplus = pickle.load(f)

    # Load x-ray model
    nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")

    return model, gradcamplusplus, nlp


# Variables for storing app persistent data
@st.cache(persist=True, allow_output_mutation=True)
def cached_variables():
    return {
        'img_idx': 0,
        'your_answer': '',
        'xray_idx': 0
    }


# Function to get gradcamed image
def get_gradcam_image(image, target_categ, gradcam):
    rgb_img = ((image - image.min()) / (image.max() - image.min()))
    input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0)

    grayscale_cam = gradcam(input_tensor=input_tensor, target_category=target_categ)

    # Overlay heatmap on an image
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization


# Classify image and get gradcam heatmap
def classify_and_gradcam(image_path, model, gradcamplusplus):
    image = io.imread(image_path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    original_image = io.imread(image_path, as_gray=False)
    original_image = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)

    transformed_image = transformations(image).unsqueeze(0)
    output_class = model(transformed_image).argmax().item()

    gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)

    if output_class == 0:
        title_text = 'Normal'
    elif output_class == 1:
        title_text = 'Benign'
    else:
        title_text = 'Malignant'

    return original_image, gradcamed_image, title_text


# Global variables
breast_root = 'C:\\Users\\Otabek Nazarov\\Desktop\\ML\\kaggle\\medtrain\\data\\breast_cancer\\'
xray_root = 'C:\\Users\\Otabek Nazarov\\Desktop\\ML\\kaggle\\medtrain\\data\\xray\\'
image_names = [
    'benign (1).png', 'malignant (1).png', 'benign (2).png', 'benign (3).png',
    'malignant (2).png', 'normal (2).png', 'normal (2).png',
]
image_xray = [
    '21.png', '24.png', '29.png'
]
texts_xray = [
    "Impression pa and lateral chest compared to through at 359 pm . \
    subcutaneous emphysema in the right chest wall has diminished slightly since removal \
    of the right pleural tube . there is still a small pocket of air and fluid , or clot \
    in the right upper chest alongside the surgical rib fracture . right lung is diffusely \
    edematous , perhaps from hilar lymphatic or venous congestion . left lung is hyperinflated \
    due to emphysema and clear of any focal abnormality . the heart is normal size . fullness in \
    the postoperative right hilus has improved since . lateral view shows persistence of an anterior \
    air and fluid collection , which on the frontal view is at the level of the third anterior interspace .",

    "Comparison is made with prior study performed a day earlier . interstitial opacities \
    in the right lung have minimally increased , likely due to edema . in the right upper hemithorax several \
    air-fluid levels are more conspicuous than in prior studies . right perihilar opacity is grossly unchanged , \
    allowing the difference in position of the patient . the left lung is clear . the right chest tube remains in \
    unchanged position . right chest wall subcutaneous emphysema has improved .",

    "Findings ap single view of the chest has been obtained with patient in sitting semi-upright \
    position . comparison is made with the next preceding similar study obtained four hours earlier during the same \
    day . again identified is status post right upper lobectomy with moderately elevated right-sided diaphragm and \
    local chest wall emphysema in the right shoulder area . no pneumothorax has developed since the preceding study , \
    and no new infiltrates are seen . impression stable chest findings as seen on portable followup examination , \
    status post right upper lobectomy ."
]


if __name__ == '__main__':
    # Load models
    classifier_model, gradcam_model, nlp = load_models(breast_root)

    # Sidebar menu
    st.sidebar.header('MedtraAIn')
    option = st.sidebar.selectbox('Select trainer', (' ', 'Chest X-Ray', 'Breast cancer'))

    # Landing page
    if option == ' ':
        title = 'MedtrAIn'
        intro_message = 'Trainer for beginner medical doctors powered by AI'
        st.markdown(f"<h1 style='text-align: center; color: white;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: white;'>{intro_message}</h3>", unsafe_allow_html=True)
        landing_img_path = 'C:\\Users\\Otabek Nazarov\\Desktop\\ML\\kaggle\\medtrain\\data\\landing.jpg'
        landing_image = io.imread(landing_img_path, as_gray=False)
        col1, col2, col3 = st.columns([0.1, 7, 0.1])
        col2.image(landing_image)

    # Breast cancer classifier page
    elif option == 'Breast cancer':
        # Load cached variables
        app_cache = cached_variables()

        # Initial text
        st.title('Breast cancer classification')
        st.markdown(
            'This trainer will assist you in learning how to classify breast cancer types. Steps are following:')
        st.markdown('**1.** Given an US breast image you first classify it yourself')
        st.markdown('**2.** Our model then gives a hint for you to reconsider your decision')
        st.markdown('**3.** You may change your choice')
        st.markdown(' ')
        st.markdown(' ')

        # Show image          
        dummy, col01, col02, col03 = st.columns([0.5, 0.5, 2, 1])

        img_path = breast_root + image_names[app_cache['img_idx']]
        image, gradcamed_image, title_text = classify_and_gradcam(img_path, classifier_model, gradcam_model)

        imageLocation = col02.empty()
        imageLocation.image(image, use_column_width=True, clamp=True)

        # Buttons for making selection
        dummy_col, benign_col, malign_col, normal_col = st.columns([0.5, 1, 1.05, 1])

        # Get the answer from the user
        with benign_col:
            if st.button('Benign'):
                imageLocation.image(gradcamed_image, use_column_width=True)
                app_cache['your_answer'] = 'Benign'

        with malign_col:
            if st.button('Malignant'):
                imageLocation.image(gradcamed_image, use_column_width=True)
                app_cache['your_answer'] = 'Malignant'

        with normal_col:
            if st.button('Normal'):
                imageLocation.image(gradcamed_image, use_column_width=True)
                app_cache['your_answer'] = 'Normal'

        # Pring out result
        dummy, col11, col12, col13 = st.columns([0.47, 1, 1.05, 1])
        if col12.button('See answer'):
            if 'benign' in img_path:
                text_color = 'green' if app_cache['your_answer'].lower() == 'benign' else 'red'
                st.markdown(f"<h5 style='text-align: center; color: {text_color};'>Your diagnosis: {app_cache['your_answer']}</h5>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: green;'>True diagnosis: Benign</h5>", unsafe_allow_html=True)
                if text_color == 'red':
                    st.markdown('A benign tumor has distinct, smooth, regular borders. \
                        A benign tumor can become quite large, but it will not invade nearby \
                        tissue or spread to other parts of your body.')
            elif 'malignant' in img_path:
                text_color = 'green' if app_cache['your_answer'].lower() == 'malignant' else 'red'
                st.markdown(f"<h5 style='text-align: center; color: {text_color};'>Your diagnosis: {app_cache['your_answer']}</h5>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: green;'>True diagnosis: Malignant</h5>", unsafe_allow_html=True)
                if text_color == 'red':
                    st.markdown('A malignant tumor has irregular borders and grows faster than a benign tumor. \
                        A malignant tumor can also spread to other parts of your body.')
            else:
                text_color = 'green' if app_cache['your_answer'].lower() == 'normal' else 'red'
                st.markdown(f"<h5 style='text-align: center; color: {text_color};'>Your diagnosis: {app_cache['your_answer']}</h5>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: green;'>True diagnosis: Normal</h5>", unsafe_allow_html=True)
                if text_color == 'red':
                    st.markdown('No visual evidence of tumor detected. \
                        Biomarkers are specific biological mechanisms and entities that may \
                        indicate existence of cancer through presence or activity. Additional research may be required.')

        # Switch to next image button
        dummy, col21, col22, col23 = st.columns([0.5, 1.13, 1, 1])
        if col22.button('Next'):
            app_cache['img_idx'] = (app_cache['img_idx'] + 1) % len(image_names)
            st.experimental_rerun()

    # Chest X-ray diagnosis page
    elif option == 'Chest X-Ray':
        app_cache = cached_variables()
        st.title('Chest X-Ray')
        st.markdown(' ')
        st.markdown(' ')

        dummy, col01, col02, col03 = st.columns([0.5, 0.5, 4, 1])

        landing_img_path = image_xray[app_cache['xray_idx']]
        landing_image = io.imread(xray_root + landing_img_path, as_gray=False)
        imageLocation = col02.empty()
        imageLocation.image(landing_image, use_column_width=True, clamp=True)

        st.text_input("Any questions", key="question")
        question = st.session_state.question

        text = texts_xray[app_cache['xray_idx']]
        answer = None
        if question:
            answer = nlp({
                "question": question,
                "context": text
            })

        if answer:
            st.text(answer)

        col21, col22, col23 = st.columns([0.5, 1, 1])

        if col23.button('Show report'):
            st.markdown(f"{text}")

        if col22.button('Next'):
            app_cache['xray_idx'] = (app_cache['xray_idx'] + 1) % len(image_xray)
            st.experimental_rerun()
