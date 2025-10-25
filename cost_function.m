function [J, grad] = cost_function(params, X, y, lambda, ...
                   input_layer_size, hidden_layer_size, ...
                   output_layer_size)

  % params -> vector containing the weights from the two matrices
  %           Theta1 and Theta2 in an unrolled form (as a column vector)
  % X -> the feature matrix containing the training examples
  % y -> a vector containing the labels (from 1 to 10) for each
  %      training example
  % lambda -> the regularization constant/parameter
  % [input|hidden|output]_layer_size -> the sizes of the three layers

  % J -> the cost function for the current parameters
  % grad -> a column vector with the same length as params
  % These will be used for optimization using fmincg

  % Extragem Theta1 (s2 x s1+1) si Theta2 (s3 x s2+1) din params
  % Transformam coloana din params in matrice cu reshape
    m = length(params);
    params1 = params(1 : hidden_layer_size * (input_layer_size + 1));
    Theta1 = reshape(params1, hidden_layer_size, input_layer_size + 1);

    params2 = params(hidden_layer_size * (input_layer_size + 1) + 1 : m);
    Theta2 = reshape(params2, output_layer_size, hidden_layer_size + 1);

  % Forward propagation
    [m, n] = size(X);
    a1 = ones(1, m);
    a1 = [a1; X'];

    z2 = Theta1 * a1;
    a2 = ones(1, m);
    a2 = [a2; sigmoid(z2)];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % Fiecare coloana din a3 are predictiile pentru clase

  % Calculam eroarea din ultimul layer si facem backpropagation
    % Transformam y in matrice
    % Coloana i din y_matrix reprezinta predictia corecta
    y_matrix = zeros(10, m);
    for i = 1 : m
      y_matrix(y(i), i) = 1;
    endfor

    % Costul (J) calculat dupa formula
    cost = (1/m) * sum(sum(-y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3)));

    % Termenul al doilea din formula lui j, termenul de regularizare
    reg = sum(sum(Theta1(:, 2 : end) .^ 2)) + sum(sum(Theta2(:, 2 : end) .^ 2));
    cost = cost + (lambda / (2*m)) * reg;

    % Calculam delta3 si delta2 cu backpropagation
    delta3 = a3 - y_matrix;

    % Derivata sigmoidului este doar sigmoid(z2) * (1 - sigmoid(z2) =
    %                                         a2 * (1 - a2)

    sigmoid_der = a2 .* (1 - a2);
    delta2 = (Theta2' * delta3) .* sigmoid_der;
    delta2 = delta2(2 : hidden_layer_size + 1, 1 : m);

  % Calculam gradientii de aceleasi dimensiuni ca Theta1 si Theta2
    gradient2 = zeros(output_layer_size, hidden_layer_size + 1);
    gradient2 = (1/m) * (gradient2 + delta3 * a2');

    gradient1 = zeros(hidden_layer_size, input_layer_size + 1);
    gradient1 = (1/m) * (gradient1 + delta2 * a1');

    gradient2(:, 2:end) = gradient2(:, 2:end) + (lambda / m) * Theta2(:, 2:end);
    gradient1(:, 2:end) = gradient1(:, 2:end) + (lambda / m) * Theta1(:, 2:end);

  % Calculm in final J si grad
    J = cost;

    % "Desfacem" gradientii din matrice in coloane
    grad = [gradient1(:); gradient2(:)];
end
