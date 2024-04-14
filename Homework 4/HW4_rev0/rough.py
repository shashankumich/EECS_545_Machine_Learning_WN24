def initialize_convnet(input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
                       hidden_dim=100, num_classes=10, dtype=np.float32):
    params = {}
    (C, H, W) = input_dim

    # Calculate the size of the output of each layer, assuming stride=1 for convolutional layers and 2 for pooling
    # Also assuming padding='valid' (no padding), hence padding=0 for calculations
    # Output size after convolution: (W-F+2P)/S + 1, for pooling it's halved due to 2x2 pooling with stride 2

    # After first conv layer
    conv1_output_size = ((H - filter_size) + 1, (W - filter_size) + 1)
    # After first pool layer
    pool1_output_size = (conv1_output_size[0] // 2, conv1_output_size[1] // 2)

    # After second conv layer
    conv2_output_size = (pool1_output_size[0] - filter_size + 1, pool1_output_size[1] - filter_size + 1)
    # After second pool layer
    pool2_output_size = (conv2_output_size[0] // 2, conv2_output_size[1] // 2)

    # Flatten size for the first fully connected layer
    flatten_size = num_filters_2 * pool2_output_size[0] * pool2_output_size[1]

    # Weight and bias initialization
    # Convolutional layers
    weight_scale_conv1 = np.sqrt(1 / (C * filter_size**2))
    weight_scale_conv2 = np.sqrt(1 / (num_filters_1 * filter_size**2))

    # Fully connected layers
    weight_scale_fc1 = np.sqrt(1 / flatten_size)
    weight_scale_fc2 = np.sqrt(1 / hidden_dim)

    params['W1'] = np.random.uniform(-weight_scale_conv1, weight_scale_conv1, (num_filters_1, C, filter_size, filter_size))
    params['W2'] = np.random.uniform(-weight_scale_conv2, weight_scale_conv2, (num_filters_2, num_filters_1, filter_size, filter_size))
    params['W3'] = np.random.uniform(-weight_scale_fc1, weight_scale_fc1, (flatten_size, hidden_dim))
    params['b3'] = np.zeros(hidden_dim)
    params['W4'] = np.random.uniform(-weight_scale_fc2, weight_scale_fc2, (hidden_dim, num_classes))
    params['b4'] = np.zeros(num_classes)

    # Convert to specified dtype
    for k, v in params.items():
        params[k] = v.astype(dtype)

    return params

# Call the function to initialize the network parameters
params = initialize_convnet()
params.keys()