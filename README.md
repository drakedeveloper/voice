AUDIO CLASSIFICATION CNN MODEL - README
=============================================

PROJECT OVERVIEW
----------------
This project implements a Convolutional Neural Network (CNN) for audio classification 
using the Google Speech Commands dataset. The model extracts MFCC (Mel-frequency cepstral 
coefficients) features from audio samples and classifies them into one of 30 different 
speech commands.


KEY FEATURES
------------
• MFCC Feature Extraction: Converts raw audio into MFCC features for effective speech 
  representation
• CNN Architecture: 3-layer convolutional neural network with batch normalization and 
  dropout
• Comprehensive Evaluation: Includes precision, recall, F1-score, and confusion matrix
• Model Saving: Supports PyTorch (.pth) and optional Keras (.h5) format
• Prediction Function: Easy-to-use function for single audio file prediction


REQUIREMENTS
------------
pip install torch torchaudio librosa numpy scikit-learn tqdm matplotlib seaborn

Optional for Keras export:
pip install tensorflow onnx onnx2keras


DATASET INFORMATION
-------------------
The model uses the Google Speech Commands dataset with the following characteristics:
• 30 different speech commands (e.g., "yes", "no", "up", "down", "left", "right", etc.)
• 1-second audio clips at 16kHz sampling rate
• Approximately 65,000 training samples


MODEL ARCHITECTURE
------------------
CNNClassifier Structure:

Input: (batch_size, 1, 40 MFCC coefficients, time_steps)

Conv Block 1:
- Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPool2d

Conv Block 2:
- Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool2d

Conv Block 3:
- Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool2d

Global Average Pooling

Classifier:
- Dropout (0.5) → Linear(128, 256) → ReLU → Dropout(0.3) → Linear(256, num_classes)

Total Parameters: ~180,000


HYPERPARAMETERS
---------------
Parameter          | Value
-------------------|--------
Sample Rate        | 16,000 Hz
Duration           | 1.0 second
MFCC Coefficients  | 40
N_FFT              | 400
Hop Length         | 200
Batch Size         | 32
Learning Rate      | 0.001
Epochs             | 20
Optimizer          | Adam
Scheduler          | StepLR (step=7, gamma=0.5)


USAGE INSTRUCTIONS
------------------

1. TRAINING THE MODEL
   The script automatically:
   - Loads the dataset from /kaggle/input/google-speech-commands/
   - Extracts MFCC features
   - Splits data into train/test (80/20)
   - Trains the CNN model
   - Displays training progress with loss and accuracy
   - Shows evaluation metrics and confusion matrix

2. MAKING PREDICTIONS
   
   For a single audio file:
   -------------------------------------------------
   # Load the saved model
   loaded_model, loaded_label_map, loaded_config = load_pytorch_model('audio_cnn_model.pth')
   
   # Predict
   result = predict_audio('path/to/audio.wav', loaded_model, loaded_label_map, device)
   print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.2f})")
   print("Top-3 predictions:")
   for cls, conf in zip(result['top3_classes'], result['top3_confidences']):
       print(f"  {cls}: {conf:.2f}")
   -------------------------------------------------

3. LOADING SAVED MODELS
   
   PyTorch format (.pth):
   -------------------------------------------------
   model, label_map, config = load_pytorch_model('audio_cnn_model.pth', device='cuda')
   -------------------------------------------------
   
   Keras format (.h5) - if saved:
   -------------------------------------------------
   model, label_map, config = load_keras_model('audio_cnn_model.h5')
   -------------------------------------------------


OUTPUT FILES
------------
After training, the following files are generated:
• /kaggle/working/audio_cnn_model.pth - PyTorch model with weights and metadata
• audio_cnn_model.h5 - Keras format (if selected)
• Training history plots
• Confusion matrix visualization


PERFORMANCE METRICS
-------------------
The model evaluation includes:
• Per-class metrics: Precision, recall, and F1-score for each of the 30 commands
• Macro averages: Average of metrics across all classes
• Weighted averages: Weighted by class support
• Confusion matrix: Visual representation of predictions vs actual labels
• Overall accuracy: Test set accuracy


SAMPLE RESULTS
--------------
After training for 20 epochs, the model achieves:

Test Loss: 0.1891
Test Accuracy: 94.61%

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

         bed       0.91      0.94      0.92       356
        bird       0.92      0.96      0.94       341
         cat       0.97      0.95      0.96       320
         dog       0.94      0.94      0.94       334
        down       0.97      0.92      0.94       484
       eight       0.90      0.97      0.93       460
        five       0.95      0.95      0.95       483
        four       0.99      0.93      0.96       478
          go       0.85      0.92      0.88       450
       happy       0.99      0.98      0.98       335
       house       0.97      0.97      0.97       346
        left       0.93      0.96      0.95       453
      marvin       0.98      0.96      0.97       351
        nine       0.97      0.96      0.96       483
          no       0.95      0.92      0.93       495
         off       0.94      0.93      0.93       497
          on       0.96      0.95      0.96       450
         one       0.99      0.94      0.96       463
       right       0.96      0.96      0.96       462
       seven       0.98      0.94      0.96       501
      sheila       0.99      0.97      0.98       360
         six       0.93      0.95      0.94       446
        stop       0.98      0.96      0.97       485
       three       0.91      0.91      0.91       497
        tree       0.92      0.89      0.90       324
         two       0.93      0.96      0.95       483
          up       0.85      0.95      0.90       512
         wow       0.98      0.95      0.97       353
         yes       0.97      0.96      0.97       471
        zero       0.98      0.96      0.97       472

    accuracy                           0.95     12945
   macro avg       0.95      0.95      0.95     12945
weighted avg       0.95      0.95      0.95     12945


============================================================
SUMMARY METRICS
============================================================
Macro Average - Precision: 0.9483, Recall: 0.9466, F1-Score: 0.9471
Weighted Average - Precision: 0.9474, Recall: 0.9461, F1-Score: 0.9464
