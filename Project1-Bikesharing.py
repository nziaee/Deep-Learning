		#sigmoid ativation function
        self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        #Forward pass hidden layer 
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        #Forward pass output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
		
        # Backward pass output error
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        #print(error.shape)
		
        # Backpropagated error terms
        output_error_term = error
        #print(output_error_term.shape)
        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term , self.weights_hidden_to_output.T)
        #print(hidden_error.shape)
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)
        #print(hidden_error_term.shape)
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        #print(hidden_outputs)
        #print(delta_weights_h_o)
        result = output_error_term * hidden_outputs
        #print(result)
        result = result.reshape(delta_weights_h_o.shape)
        #print(result)
        delta_weights_h_o += result

############################
# hyperparameters
############################
iterations = 4000
learning_rate = 0.6
hidden_nodes = 15
output_nodes = 1
