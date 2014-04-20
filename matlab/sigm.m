function X = sigm(P)
  X = (1 + tanh(P/2)) / 2;
  %X = 1./(1+exp(-P));
end