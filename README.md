# BrainTumorDetection
Project - 
Identifiying 3 different (Glioma, Meningioma, Pituitary Adenoma) types of tumor based on MRI scan images of patients.


### File Structure 
1. GUI_Master_old.py - frontend of the interface.
2. caps.py - backend training of the capsule model.

### How to Run
1. If the caps_network.h5 file is already present then you can directly run the 'GUI_Master_old.py' file [ training already performed ]
2. Otherwise, run 'caps.py' --> 'model_caps.py', which will begin training the images.


### Note - 
1. Training and Testing data are not available. You can find them on Kaggle.
2. Change the location of training and testing data in the folder appropriately.
3. caps_network.h5 - The saved model weights from prior training. 
