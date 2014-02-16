function layers = initact(layers, data)

layers{1}.a = data;
layers{1}.a(-layers{1}.eps < layers{1}.a & layers{1}.a < layers{1}.eps) = 0;

end