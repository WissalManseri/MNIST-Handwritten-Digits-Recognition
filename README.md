# MNIST Handwritten Digits Recognition
This program uses a convolutional neural network (CNN) to recognize handwritten digits from the MNIST dataset. 
The CNN is implemented using the Keras library, and the program is written in Python.


# Requirements
To run this program, you need to have the following installed:

      Python 3.6 or higher
      Keras 2.4 or higher
      NumPy
      Matplotlib
      
You can install these dependencies using pip by running:


       pip install -r requirements.txt
       
# Usage
To train the model and save it to a file, run:


        
        python train_model.py
To test the model on a single image, run:


        python test_model.py <image_path>
        
Where <image_path> is the path to the image file.

To test the model on the entire test set, run:


         python evaluate_model.py
         
This will print out the accuracy of the model on the test set.

# Credits
This program was  was inspired from the :

 The Data Science Course 2023: Complete Data Science Bootcamp (udemy)


https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/learn/lecture/15088966?start=0#overview

# More informations :  Neural Network Architecture

The CNN used in this program has the following architecture:

            Input layer (28 x 28 grayscale image)
            Convolutional layer (32 filters, kernel size 3 x 3, ReLU activation)
            Max pooling layer (pool size 2 x 2)
            Convolutional layer (64 filters, kernel size 3 x 3, ReLU activation)
            Max pooling layer (pool size 2 x 2)
            Flatten layer
            Dense layer (128 neurons, ReLU activation)
            Dropout layer (dropout rate 0.5)
            Output layer (10 neurons, softmax activation)
The model is trained using the Adam optimizer and the categ
orical cross-entropy loss function. During training, the model uses early stopping and checkpointing to prevent overfitting and save the best weights.

![MNIST Handwritten Digits Recognition PH1](https://user-images.githubusercontent.com/89531771/225841409-4d916e33-384e-458c-85dc-ab63e8fa3578.png)


<img src="https://user-images.githubusercontent.com/89531771/225841418-4aa0e319-84a7-4316-808b-9aa0bbf16d9a.jpg" alt="MNIST Handwritten Digits Recognition PH2" width="1000" >




   # Training
To train the model, we first load the MNIST dataset using Keras. We then preprocess the data by scaling the pixel values to the range [0, 1] and converting the labels to one-hot encoded vectors. We then split the data into training and validation sets.

The model is trained using mini-batch gradient descent with a batch size of 128 and a learning rate of 0.001. We train for a maximum of 20 epochs, but use early stopping if the validation loss does not improve for 5 epochs.

   # Testing
To test the model, we load the saved weights and evaluate the accuracy on the test set. We also visualize some of the misclassified images using Matplotlib.

# License
This program is licensed under the MIT License. See the LICENSE file for details.


