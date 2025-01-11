from sklearn.neural_network import MLPClassifier
# Let us now create a neural network model to learn from train data
# We shall build a single hidden layer with 50 nodes. 
nn_model = MLPClassifier(hidden_layer_sizes=(50))
# The fit method initiates the learning process. When its execution completes, the model is learnt
nn_model.fit(X_train,mnist_train.iloc[:,0])
# Now that we have a model, lets get it to predict the value of the first digit in the test data
print(nn_model.predict(mnist_test.iloc[0:1,]))
# output
# [2]
