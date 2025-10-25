function [classes] = predict_classes(X, weights, ...
                  input_layer_size, hidden_layer_size, ...
                  output_layer_size)
  % X -> the test examples for which the classes must be predicted
  % weights -> the trained weights (after optimization)
  % [input|hidden|output]_layer_size -> the sizes of the three layers

  % classes -> a vector with labels from 1 to 10 corresponding to
  %            the test examples given as parameter

  % Extragem Theta_i din weights, la fel ca la cost_function
    m = length(weights);
    weights1 = weights(1 : hidden_layer_size * (input_layer_size + 1));
    Theta1 = reshape(weights1, hidden_layer_size, input_layer_size + 1);

    weights2 = weights(hidden_layer_size * (input_layer_size + 1) + 1 : m);
    Theta2 = reshape(weights2, output_layer_size, hidden_layer_size + 1);

  % Forward propagation, la fel ca la cost_function
    [m, n] = size(X);
    a1 = ones(1, m);
    a1 = [a1; X'];

    z2 = Theta1 * a1;
    a2 = ones(1, m);
    a2 = [a2; sigmoid(z2)];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

  % Calculam predictiile cu ajutorul neuronului din ultimul layer
    classes = zeros(m, 1);
    for i = 1 : m
      [mx, imx] = max(a3(:, i));
      classes(i) = imx;
    endfor
end
