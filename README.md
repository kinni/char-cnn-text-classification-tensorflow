# char-cnn-text-classification-tensorflow

Simple Convolutional Neural Network (CNN) for text classification at character level. In this project we will implement a Chinese Movie Sentiment (Positive/Negative) Classifier with CNN using TensorFlow. A manually labeled small corpus (HK movie reviews) has been uploaded for you to try out the network.  

Mostly reused code from https://github.com/dennybritz/cnn-text-classification-tf which was posted by Denny Britz. Check out his great blog post on CNN classification http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/.

Things are modified from the original author:
- modified the data helpers to support Chinese to train the embeddings at character level
- added Tensorboard visualization for embeddings
- added `sample.py` which will generate prediction output
- added `serve.py` which will serve the classifier via Flask

### Requirements
- Tensorflow 1.0 or up

### Basic Usage
To train with the Chinese movie reviews dataset, run:
`python train.py`

To visualize results and embeddings, run:
`tensorboard --logdir ./runs/1492654198/summaries/`

To predict from a trained model, run with checkpoint_dir argument
` python sample.py --checkpoint_dir=./runs/1492654198/checkpoints/ --text="套戲好鬼悶"`

To serve the model via a Flask API, run:
`python serve.py --checkpoint_dir=./runs/1492656039/checkpoints/`
then send a HTTP POST request to `http://localhost:5000/predict` with the following json body:
`{"text": "套戲好鬼悶"}`

### TODO
Some updates will be published soon.
- Train the embeddings at word level using jieba's tokenizer or other pre-trained word embeddings
- Use high level APIS such as `tf.layers` or `tf.keras`(will be available in TensorFLow 1.2)