function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  % X -> the loaded dataset with all training examples
  % y -> the corresponding labels
  % percent -> fraction of training examples to be put in training dataset

  % X_[train|test] -> the datasets for training and test respectively
  % y_[train|test] -> the corresponding labels

  % Example: [X, y] has 1000 training examples with labels and percent = 0.85
  %           -> X_train will have 850 examples
  %           -> X_test will have the other 150 examples

  % Impartim in date de antrenare si testare, in mod aleatoriu
  [m, n] = size(X);

  % Se permuta linii aleatoriu din X si y
  r = randperm(m);
  X_rand = X(r, :);
  y_rand = y(r, :);

  m_train = percent * m;

  X_train = X_rand(1 : m_train, :);
  y_train = y_rand(1 : m_train);

  X_test = X_rand(m_train + 1 : m, :);
  y_test = y_rand(m_train + 1 : m);
end
