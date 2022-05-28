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


# Image transformations for model
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


@st.cache(allow_output_mutation=True, hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()})
def load_models(root='C:\\Users\\Otabek Nazarov\\Desktop\\ML\\kaggle\\med_train\\data\\breast_cancer\\'):
    # C:\Users\Otabek Nazarov\Desktop\ML\kaggle\med_train\data\breast_cancer\breast_tumor_model.pkl
    # Load DenseNet-121 model
    with open(root+'breast_tumor_model.pkl', 'rb') as f:
        model = pickle.load(f)

    model.eval()

    # Load gradcam for XAI
    with open(root+'gradcam.pkl', 'rb') as f:
        gradcamplusplus = pickle.load(f)

    return model, gradcamplusplus

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def cached_variables():
    return {'img_idx': 0}


# Function to get gradcamed image
def get_gradcam_image(image, target_categ, gradcam):
    rgb_img = ((image - image.min()) / (image.max() - image.min()))
    input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0)

    grayscale_cam = gradcam(input_tensor=input_tensor, target_category=target_categ)

    # Overlay heatmap on an image
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization


def classify_and_gradcam(image_path, model, gradcamplusplus):
    image = io.imread(image_path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)

    original_image = io.imread(image_path, as_gray=False)

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
root='C:\\Users\\Otabek Nazarov\\Desktop\\ML\\kaggle\\med_train\\data\\breast_cancer\\'
image_names = [
    'benign (1).png', 'benign (2).png', 'benign (3).png', 'malignant (1).png', 
    'malignant (2).png', 'normal (2).png', 'normal (2).png',
    ]



if __name__ == '__main__':
    # Load models
    classifier_model, gradcam_model = load_models()

    # Sidebar menu
    st.sidebar.header('MedtraAIn')
    st.sidebar.subheader('Trainer for beginner medical practitioners')
    option = st.sidebar.selectbox('Select trainer', (' ', 'Chest X-Ray', 'Breast cancer'))

    if option == ' ':
        print('nothing')

    elif option == 'Breast cancer':
        app_cache = cached_variables()
        # Initial text
        st.title('Breast cancer classification')
        st.markdown('This trainer will assist you in learning how to classify breast cancer types. Steps are following:')
        st.markdown('**1.** Given an US breast image you first classify it yourself')
        st.markdown('**2.** Our model then gives a hint for you to reconsider your decision')
        st.markdown('**3.** You make your final choice')
        st.markdown(' ')
        st.markdown(' ')
        
        # Show image          
        col1, col2, col3 = st.columns([0.5,2,1])
        print(app_cache)
        img_path = root + image_names[app_cache['img_idx']]
        image, gradcamed_image, title_text = classify_and_gradcam(img_path, classifier_model, gradcam_model)
        
        imageLocation = col2.empty()
        imageLocation.image(image, use_column_width=True, clamp=True)

        # Buttons for making selection
        benign_col, malign_col, normal_col = st.columns(3)

        with benign_col:
            if st.button('Benign'):
                imageLocation.image(gradcamed_image, use_column_width=True)

        with malign_col:
            if st.button('Malignant'):
                imageLocation.image(gradcamed_image, use_column_width=True)

        with normal_col:
            if st.button('Normal'):
                imageLocation.image(gradcamed_image, use_column_width=True)
        
        # Next button
        col11, col12, col13 = st.columns([1.13,1,1])
        if col12.button('Next'):
            app_cache['img_idx'] = (app_cache['img_idx'] + 1) % len(image_names)


        # st.markdown(' ')
        # st.markdown(' ')
        # st.markdown('Statistics:')
        # st.markdown(f'1st attempt: {2}/{3}, 2nd attempt {3}/{3}')
        # st.markdown(f'Benign: {1}/{1}, Malignant {1}/{1}, Normal {1}/{1}')
    
    elif option == 'Chest X-Ray':
        print('cxr')