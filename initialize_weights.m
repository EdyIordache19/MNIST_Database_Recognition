function [matrix] = initialize_weights(L_prev, L_next)
  % L_prev -> the number of units in the previous layer
  % L_next -> the number of units in the next layer

  % matrix -> the matrix with random values

  % Calculam epsilon_0
  epsilon = sqrt(6) / sqrt(L_prev + L_next);

  % Formula pentru a genera o matrice cu valori random intre a si b
  % La noi trebuie generate intre -epsilon_0 si epsilon_0
  a = -epsilon;
  b = epsilon;
  matrix = (b - a) .* rand(L_next, L_prev + 1) + a;
end
