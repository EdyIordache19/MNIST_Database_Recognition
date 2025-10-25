## MNIST 101
- `load_dataset(path)` - Parses a `.mat` file and loads the `X` and `y` variables, that
    it returns. `X` is a matrix, that stores the pixel values for a `20 x 20` photo of a
    number on each line, while `y` is a column with the correct prediction for the number.
- `split_dataset(X, y, percent)` - Generates two datasets, one for training, and the other
    one for testing the model. The lines from `X` and `y` are randomized, and the first
    part are put in `X_train` and `y_train`, while the second in `X_test` and `y_test`.
- `initialize_weights(L_prev, L_next)` - It initializes the weights from the formula in
    the support PDF. The weights are generated randomly, in the interval (-epsilon, epsilon),
    where epsilon is `sqrt(6) / sqrt(L_prev + L_next)`.
- `cost_function(params, X, y, lambda, input_layer_size, hidden_layer_size, output_layer_size)` -
    Calculates the cost function, according to the formulas from the PDF and trains the weigths.
    Firstly, the initial weights are taken from the `params` vector, and made into `Theta1`
    and `Theta2` matrices, of dimensions described. Then, I forward propragate, to calculate
    the predictions from the current weights, and the cost. Then, I calculate `delta3` and
    `delta2` using back propagation, and the gradients accordingly. Finally, the cost function
    `J`, is just the cost calculated, and the `grad` is just `Gradient1` and `Gradient2`, in
    column form.
- `predict_classes(X, weights, input_layer_size, hidden_layer_size, output_layer_size)` - It
    makes predictions based on the weights and returns these predictions. The same as the last
    function, builds `Theta1` and `Theta2` matrices from the `weights` column and forward
    propagates to calculate the predictions. The prediction are in a matrix form, each column
    having the probabilities for each number. It takes the maximum of each column and updates
    the classes column.