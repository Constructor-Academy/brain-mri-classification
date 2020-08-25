import os
import sys
from pathlib import Path

import tensorflow as tf

# Paths
#
# use this for jupyter notebook
# project_dir = Path(os.getcwd()).parent

# use this for python file
src_dir = Path(__file__).parent.absolute()
sys.path.insert(0, src_dir)

project_dir = Path(src_dir).parent.absolute()

mriclassifier_path = '/Users/norbiorb/Data/git/propulsion/FinalProject/mri-classifier'

print(f'project dir: {project_dir}')
data_dir = os.path.join(mriclassifier_path, 'data')
images_dir = os.path.join(mriclassifier_path, 'images')
models_dir = os.path.join(mriclassifier_path, 'models')



# function to preprocess loaded images, dependent on the base network model which is by defaulr resnet50
preprocess_input = tf.keras.applications.resnet.preprocess_input

siam_model_perspective_path = os.path.join(models_dir, 'resnet50_siamese_binary_crossentropy_120.h5')
embedding_model_perspective_path = os.path.join(models_dir, 'resnet50_siamese_base_network_120.h5')

# col = 'perspective'
# col = 'sequence'
# col = 'perspective-sequence'
df_config = {
    'cls_col': 'perspective',
    'name_col': 'image-name'
}
