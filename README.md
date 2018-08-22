# Text_Classifier_LSTM_Keras
A tool for text classification using LSTM layer and now CNN layers.

Each instance of the Classifier class allows us to classify some texts
according to sample folder which contains each category to classify.

There is a file "language_detector" which contains a save of an instance
which classify five languages: German, English, Spanish, French and Dutch.

==== PARAMETERS ====

You can choose the size of data batches loaded in memory (by default 50.0 MB)
with the parameter data_size_max. It overcomes memory issues.

Saving a Classifier instance save weights in the same file and in a separated h5 file,
so h5 file isn't required to load weights of a loaded Classifier instance.

==== REQUIREMENTS ====
- Python 3.5+
- Keras 2.x (deep learning API)
- Keras backend: Theano / Tensorflow (deep learning engine)
- h5py (for saving models)
- Numpy (matrix calculations)
