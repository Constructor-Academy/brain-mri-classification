


## Scripts for Siamese Network

### siamese_config.py
Contains configuration needed to run a script. Imported by the scripts.
However training specific configuration is done in `siamese_network.py` which runs the training.

### image_pair_generator.py
contains `ImagePairGenerator` class, which generates batches of image pairs from image names
coming from a pandas dataframe. It also augments the images according to the configuration 
given as constructor argument or as default configuration.
This batches of pairs are used to feed the siamese network inputs.

### siamese_model.py
Contains the `SiameModel` class used to create siamese network and its mbase network. The basenetwork can be used to create embeddings.

`python siamese_model.py` called on commandline builds the model and shows its summary as output.

### siamese_network.py
Contains the configuration and code for training the model defined in `siamese_model.py`.

`python siamese_network.py` called on the commandline starts the training of the model with the given configuration. The trained model is saved at the end of training
and after training the history and the model(s) are stored in the filesystem.

### siamese_predict.py 
Contains `SiamesePredictor` which can be used to create predictions from trained `SiameseModel`s.
It also contains an `ImageLoader` which can be used to load images and also `dcim` images.

`python siamese_predict.py` called on the commandline expects to find pretrained networks 
(siamese network and its basenetwrok) at the locations defined in `siamese_config.py`.
It plots the similarity of a sample image compared to randomly sampled class members.

### siamese_util.py 
Contains helper functions for sampling pairs and helper functions to create `.tsv` files which can be used to 
visualize embeddings created by the base network of the `SiameseModel`. It can be shown e.g. in the [online tensorflow projector](https://projector.tensorflow.org/).



