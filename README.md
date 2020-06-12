# Optimization for Machine Learning - CS-439 
# Mini-Porject - Studying generalisation in first-order optimisation algorithms

This project compares generalisation in different first-order Optimisation algorithms such as Adam, SGD and RMSprop. 
In particular, this study expores PAC-Bayes theory and variance of gradients as complexity measures to compare generalisation in different optimization algorihtms.

## Project Structure

The project is structured as follows:

    .
    ├── utils.py                      # Implements training, testing and different dataloaders
    ├── measures.py                   # Implements different complexity measures
    ├── model.py                      # Implements architecture of the model
    ├── avg_adam_norm_sharpness.ipynb # main file running different optimisation algorithms
    ├── .py        # Functions for preprocessing
    ├──Norm_Sharpness
                     ├──avg_adam_norm_sharpness.ipynb  # Norm Sharpness for Adam
                     ├──avg_rms_norm_sharpness.ipynb   # Norm Sharpness for RMSprop
                     ├──avg_sgd_norm_sharpness.ipynb   # Norm Sharpness for SGD
    ├──Variance_of_Gradients
                     ├──avg_adam_var_of_grad.ipynb     # Var of grad for Adam for 3 different learning rates
                     ├──avg_rms_var_of_grad.ipynb      # Var of grad for RMSprop for 3 different learning rates
                     ├──avg_sgd_var_of_grad.ipynb      # Var of grad for SGD for 3 different learning rates           
    └── README.md               # README

## Dependencies

The only required dependencies are:

```
NumPy
Matplotlib
Pytorch
```

## Running

You can run `avg_adam_norm_sharpness.ipynb` either on colab or any other platform(with dependecies installed)to see results for norm and sharpness measures. You can experiment with different sigma values or different model hyperparameters by changing values accordingly in the notebook. 
You can run variance of gradients and norm shaprness measures for other algorithms in a similar way. 
Note** Don't forget to include the utility files that implement complexity measures and dataloaders etc. 
The Pytorch dataloader function automatically downloads MNIST data if data file is not found at the desired location.

## Authors

* [Manjot Singh] (manjot.singh@epfl.ch)
* [Kshiteej Jitesh Sheth] (kshiteej.sheth@epfl.ch)
