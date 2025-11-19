function net = zhang_forward_relog(net, input_data)
    % Assign input data to the first layer
    % This just stores the input without any processing
    net.activations{1} = input_data;  
    net.v{1} = input_data;  

    % Forward pass for hidden layers (ReLog activation function)
    for layer = 2:net.n_layers - 1  
        % Compute the weighted sum of inputs (plus bias)
        net.v{layer} = (net.W{layer} .* net.connectivity{layer}) * net.activations{layer - 1} + net.b{layer};  
        
        % Apply ReLog activation: log(1 + |x|) but keeps sign information
        net.activations{layer} = max(0, sign(net.v{layer}) .* log(1 + abs(net.v{layer})));  
    end

    % Output layer (final layer, fully connected to both previous layers)
    final_layer = net.n_layers;  
    net.v{final_layer} = (net.W{final_layer} .* net.connectivity{final_layer}) * ...
                          [net.activations{final_layer-2}; net.activations{final_layer-1}] + net.b{final_layer};  

    % No activation function in the output layer, just raw values
    net.activations{final_layer} = net.v{final_layer};  

end