function [X, y] = load_dataset(path)
  % path -> a relative path to the .mat file that must be loaded

  % X, y -> the training examples (X) and their corresponding labels (y)

  % Facem load la fisierului .mat si se extrag automat X si y
  load(path);
end
