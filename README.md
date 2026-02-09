# Workflow description
Input data must be a .jpg image, which will be automatically processed and passed to the first layer of the model. 
This will classify the image depending on whether a tumor is detected or not, and in positive case it predicts which type.
Next, after prediction, second model is applied to segmentate and indicate on the image where the detected tumor is. 
Output data is the tumor label and the mask remarking where the tumor has been detected.

# Content
Provided .py codes stand for the training of both models. To test entire composed model, use eval_single_image.py.
.pdf document contains full explanation of the process and implementation.
