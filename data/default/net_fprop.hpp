void printTensorSize(THTensor* input) {
    for(int i = 0; i < input->nDimension; i++) {
        cout << input->size[i] << " ";
    }
    cout << endl;
}

void printTensorContents(THTensor* input) {
    real* layer_data = THTensor_(data)(input);
    long
    sf = input->stride[0],
    sy = input->stride[1],
    sx = input->stride[2];
    printTensorSize(input);

    for (int i = 0; i < input->size[0]; ++i){
        for (int y = 0; y < input->size[1]; ++y){
        for (int x = 0; x < input->size[2]; ++x){
            cout << layer_data[i*sf+y*sy+x*sx] << " ";
        }
        cout << endl;
        }
    }
}

THTensor* fprop1(THTensor* input, int net_idx, int zero_layer,
                 int zero_perc, int mask_layer) {

    std::unordered_map<string,int> layer_weight_map = {
        {"01", 1},
        {"02", 4},
        {"03", 8},
        {"04", 11},
        {"05", 14},
        {"06", 17},
        {"07", 19},
        {"08", 21},
        {"11", 1},
        {"12", 4},
        {"13", 8},
        {"14", 11},
        {"15", 14},
        {"16", 17},
        {"17", 20},
        {"18", 22},
        {"19", 24}
    };

    if(zero_layer > 0) {
        srand(time(NULL));
        string key = to_string(net_idx) + to_string(zero_layer);
        if(layer_weight_map.count(key) == 0) {
            cout << "Invalid key given to layer_weight_map" << endl;
            exit(0);
        }

        int w_i = layer_weight_map[key];
        float drop_percentage = float(zero_perc) / 100.0f;

        //cout << "Weight to zero out: " << w_i << endl;
        auto cur_weight = weights[w_i];
        auto cur_bias = bias[w_i];

        real* weight_data = THTensor_(data)(cur_weight);
        int num_weights = 1;
        //cout << "Weight dim: " << cur_weight->nDimension << endl;
        for(int i = 0; i < cur_weight->nDimension; i++) num_weights *= cur_weight->size[i];

        //cout << "Number of weights: " << num_weights << endl;
        //cout << "Drop percentage: " << drop_percentage << endl;
        for(int i  = 0; i < num_weights; i++) {
            float r = float(rand()) / float(RAND_MAX);

            if(r < drop_percentage) {
                weight_data[i] = 0;
            }
        }

        real* bias_data = THTensor_(data)(cur_bias);
        int num_bias = 1;
        //cout << "Bias dim: " << cur_bias->nDimension << endl;
        for(int i = 0; i < cur_bias->nDimension; i++) num_bias *= cur_bias->size[i];

        //cout << "Number of bias: " << num_bias << endl;
        for(int i  = 0; i < (drop_percentage * num_bias); i++) {
            float r = float(rand()) / float(RAND_MAX);

            if(r < drop_percentage) {
                bias_data[i] = 0;
            }
        }
    }



    if (net_idx == 0) {
        if(mask_layer >= 0 && mask_layer != 5) {
            fprintf(stderr, "Attempting to mask layer %d "
                            "in fast model. Only layer 5 masking is allowed\n", mask_layer);
            exit(0);
        }
    // layer 1
    Normalization_updateOutput(input, 118.380948, 61.896913, outputs[0]);
    SpatialConvolution_updateOutput(outputs[0], 4, 4, weights[1], bias[1], outputs[1]);
    Threshold_updateOutput(outputs[1], 0.000000, 0.000001, outputs[2]);
    SpatialMaxPooling_updateOutput(outputs[2],2,2,2,2,weights[3], outputs[3]);

    // layer 2
    SpatialConvolution_updateOutput(outputs[3], 1, 1, weights[4], bias[4], outputs[4]);
    Threshold_updateOutput(outputs[4], 0.000000, 0.000001, outputs[5]);
    SpatialMaxPooling_updateOutput(outputs[5],2,2,2,2,weights[6], outputs[6]);

    // layer 3
    SpatialZeroPadding_updateOutput(outputs[6], 1, 1, 1, 1, outputs[7]);
    SpatialConvolution_updateOutput(outputs[7], 1, 1, weights[8], bias[8], outputs[8]);
    Threshold_updateOutput(outputs[8], 0.000000, 0.000001, outputs[9]);

    // layer 4
    SpatialZeroPadding_updateOutput(outputs[9], 1, 1, 1, 1, outputs[10]);
    SpatialConvolution_updateOutput(outputs[10], 1, 1, weights[11], bias[11], outputs[11]);
    Threshold_updateOutput(outputs[11], 0.000000, 0.000001, outputs[12]);
    THTensor* layer6_input = outputs[12];

    // layer 5
    if(mask_layer != 5){
        SpatialZeroPadding_updateOutput(outputs[12], 1, 1, 1, 1, outputs[13]);
        SpatialConvolution_updateOutput(outputs[13], 1, 1, weights[14], bias[14], outputs[14]);
        Threshold_updateOutput(outputs[14], 0.000000, 0.000001, outputs[15]);
        layer6_input = outputs[15];
    }
    SpatialMaxPooling_updateOutput(layer6_input,2,2,2,2,weights[16], outputs[16]);

    // layer 6
    SpatialConvolution_updateOutput(outputs[16], 1, 1, weights[17], bias[17], outputs[17]);
    Threshold_updateOutput(outputs[17], 0.000000, 0.000001, outputs[18]);

    // layer 7
    SpatialConvolution_updateOutput(outputs[18], 1, 1, weights[19], bias[19], outputs[19]);
    Threshold_updateOutput(outputs[19], 0.000000, 0.000001, outputs[20]);

    // layer 8 (final)
    SpatialConvolution_updateOutput(outputs[20], 1, 1, weights[21], bias[21], outputs[21]);
    return outputs[21];
    }
    if (net_idx == 1) {
        if(mask_layer >= 0 && mask_layer != 4 && mask_layer != 6){
            fprintf(stderr, "Attempting to mask layer %d "
                    "in accuract model. Only layer 4 or 6 masking is allowed\n", mask_layer);
            exit(0);
        }

    // layer 1
    Normalization_updateOutput(input, 118.380948, 61.896913, outputs[0]);
    SpatialConvolution_updateOutput(outputs[0], 2, 2, weights[1], bias[1], outputs[1]);
    Threshold_updateOutput(outputs[1], 0.000000, 0.000001, outputs[2]);
    SpatialMaxPooling_updateOutput(outputs[2],3,3,3,3,weights[3], outputs[3]);

    // layer 2
    SpatialConvolution_updateOutput(outputs[3], 1, 1, weights[4], bias[4], outputs[4]);
    Threshold_updateOutput(outputs[4], 0.000000, 0.000001, outputs[5]);
    SpatialMaxPooling_updateOutput(outputs[5],2,2,2,2,weights[6], outputs[6]);

    // layer 3
    SpatialZeroPadding_updateOutput(outputs[6], 1, 1, 1, 1, outputs[7]);
    SpatialConvolution_updateOutput(outputs[7], 1, 1, weights[8], bias[8], outputs[8]);
    Threshold_updateOutput(outputs[8], 0.000000, 0.000001, outputs[9]);
    THTensor* layer5_input = outputs[9];

    // layer 4
    if(mask_layer != 4) {
        SpatialZeroPadding_updateOutput(outputs[9], 1, 1, 1, 1, outputs[10]);
        SpatialConvolution_updateOutput(outputs[10], 1, 1, weights[11], bias[11], outputs[11]);
        Threshold_updateOutput(outputs[11], 0.000000, 0.000001, outputs[12]);
        layer5_input = outputs[12];
    }

    // layer 5
    SpatialZeroPadding_updateOutput(layer5_input, 1, 1, 1, 1, outputs[13]);
    SpatialConvolution_updateOutput(outputs[13], 1, 1, weights[14], bias[14], outputs[14]);
    Threshold_updateOutput(outputs[14], 0.000000, 0.000001, outputs[15]);
    THTensor* layer7_input = outputs[15];

    // layer 6
    if(mask_layer != 6) {
        SpatialZeroPadding_updateOutput(outputs[15], 1, 1, 1, 1, outputs[16]);
        SpatialConvolution_updateOutput(outputs[16], 1, 1, weights[17], bias[17], outputs[17]);
        Threshold_updateOutput(outputs[17], 0.000000, 0.000001, outputs[18]);
        layer7_input = outputs[18];
    }
    SpatialMaxPooling_updateOutput(layer7_input,3,3,3,3,weights[19], outputs[19]);

    // layer 7
    SpatialConvolution_updateOutput(outputs[19], 1, 1, weights[20], bias[20], outputs[20]);
    Threshold_updateOutput(outputs[20], 0.000000, 0.000001, outputs[21]);

    // layer 8
    SpatialConvolution_updateOutput(outputs[21], 1, 1, weights[22], bias[22], outputs[22]);
    Threshold_updateOutput(outputs[22], 0.000000, 0.000001, outputs[23]);

    // layer 9 (final)
    SpatialConvolution_updateOutput(outputs[23], 1, 1, weights[24], bias[24], outputs[24]);
    return outputs[24];
    }
}
