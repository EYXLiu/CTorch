# CTorch Library
* Made with C++ and Object Oriented Programming principles and the Factory Design Pattern
* Fully supported tensor and type classes with automatic memory management
* Implmented both forward and backwards propagation for Module containers and the Linear layer
* Implemented gradient accumulation, explained below
* Activation functions such as ReLU and Softmax
* Implemented matrix related functions such as dot product, matrix multiplication, addition, Hadamard multiplication, and more
* Uses RAII principles such as unique pointers and clean memory usage

# Setup
* Write model or test code in main.cc
* In the terminal, run make
* Run the executable (In the terminal, run './ctorch')

# Gradient Accumulation
Initially, backpropagation was implemented as a per CTensor implementation where it would propagate backwards and adjust the weights as it went, due to personal theory and interpretation. However, learning about gradient accumulation and the benefits, eg. for multicore learning, batch processing, optimizing using chain rule, allowing for gradient scaling and clipping (moreso for CNNs), etc. 

# Current working modules
* Datatypes
* Module containers
* Linear layer

# Work in progress
* Convolutional Layers
* Multi-shape Transposition and Summation
* non-Linear Activations
* Normalization
* Pooling Layers
