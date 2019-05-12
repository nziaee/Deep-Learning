
		#sigmoid ativation function
        self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.

        #Forward pass hidden layer 
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        #Forward pass output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = self.activation_function(final_inputs) # signals from final output layer

        # Backward pass output error
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # Backpropagated error terms
        output_error_term = error * final_outputs * (1-final_outputs)

        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += self.lr * hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += self.lr *  hidden_outputs * output_error_term.T

####################################
# hyperparameters 
####################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
