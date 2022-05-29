# MedtrAIn - AI powered medical trainer
## Prepared for *TumarisHack* by team *From MADE to top*


### Running the program
1. Create new python environment
```
conda env create -f requirements.yml
```
2. Place [contents](https://drive.google.com/file/d/1h2KHLc_CGGMPvWkZXZgcWKscW8WF8vZp/view?usp=sharing) in [data](data/) folder
3. Run the following command
```
streamlit run medtrain.py
```


### Main libries used for software
- pytorch
- pytorch-gradcam
- transformers (HuggingFace)
- streamlit
Full list available [here](requirements.yml)


### Datasets
[Breast cancer ultrasound images](https://www.kaggle.com/code/tanvirrahmanornob/breast-cancer-detection/data). <br>
[Chest X-Ray reports and images](https://physionet.org/content/mimic-cxr/2.0.0/)