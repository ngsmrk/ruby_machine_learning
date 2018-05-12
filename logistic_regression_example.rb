require 'liblinear'

# train
model = Liblinear.train(
  { solver_type: Liblinear::L2R_LR },   # parameter
  [-1, -1, 1, 1],                       # labels (classes) of training data
  [[-2, -2], [-1, -1], [1, 1], [2, 2]], # training data
  -1, # bias
)
# predict
(-1..1).step(0.1).each do | i |
  rounded = i.round(1)
  p rounded
  puts "Prediction: #{Liblinear.predict(model, [rounded, rounded])}"
end
