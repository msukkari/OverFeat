#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include "tools/ppm.hpp"
#include "overfeat.hpp"
using namespace std;

int main(int argc, char* argv[]) {
  // read arguments
  if (argc < 2) {
    fprintf(stderr, "Missing argument : path to weight file\n");
      exit(0);
  }
  if (argc < 3) {
    fprintf(stderr, "Missing argument : (number of top classes | -1)\n");
    exit(0);
  }
  if (argc < 4) {
    fprintf(stderr, "Missing argument : network idx\n");
    exit(0);
  }
  if (argc < 5) {
      fprintf(stderr, "Missing argument : mask layer\n");
      exit(0);
  }
  int nTopClasses = atoi(argv[2]);
  int net_idx = atoi(argv[3]);
  int mask_layer = atoi(argv[4]);
  int zero_layer = atoi(argv[5]);
  int zero_perc = atoi(argv[6]);
  int feature_layer;
  if (nTopClasses <= 0) {
    if (argc < 6) {
      fprintf(stderr, "Missing argument : output feature layer\n");
      exit(0);
    } 
    feature_layer = atoi(argv[4]);
  }

  //cout << "mask layer: " << mask_layer << endl;
  //cout << "zero_layer: " << zero_layer << endl;
  //cout << "zero_perc: " << zero_perc << endl;

  // initializes overfeat
  //cout << "INIT OVERFEAT" << endl;
  overfeat::init(argv[1], net_idx);
  
  THTensor* input_raw = THTensor_(new)();
  THTensor* input = THTensor_(new)();
  THTensor* probas = THTensor_(new)();

  while(readPPM(stdin, input_raw)) {
    assert(input_raw->size[0] == 3); //input must be rgb

    if (nTopClasses > 0) { // print top classes
      // crop image to make it square
      int rw = input_raw->size[2], rh = input_raw->size[1];
      int dstdim = min(rh, rw);
      THTensor_(resize3d)(input, 3, dstdim, dstdim);
      long
      sr0 = input_raw->stride[0],
      sr1 = input_raw->stride[1],
      sr2 = input_raw->stride[2],
      s0 = input->stride[0],
      s1 = input->stride[1],
      s2 = input->stride[2];
      int xoffset = 0, yoffset = 0;
      if (rh < rw) {
	      xoffset = (rw - dstdim)/2;
      } else {
	      yoffset = (rh - dstdim)/2;
      }
      real* data_raw = THTensor_(data)(input_raw);
      real* data = THTensor_(data)(input);
      for (int c = 0; c < 3; ++c) 
	      for (int i = 0; i < dstdim; ++i)
	        for (int j = 0; j < dstdim; ++j)
	          data[s0*c + s1*i + s2*j] = data_raw[sr0*c + (i+yoffset)*sr1 + (j+xoffset)*sr2];
      
      // classification
      THTensor* output = overfeat::fprop(input, zero_layer, zero_perc, mask_layer);
      if ((output->size[1] != 1) || (output->size[2] != 1)) {
	      cerr << "Can only determine class if the output is 1x1. Reduce input size" << endl;
	      exit(0);
      }
      output->nDimension = 1;
      overfeat::soft_max(output, probas);
      vector<pair<string, float> > top_classes = overfeat::get_top_classes(probas, nTopClasses);
      
      // print output
      for (int i = 0; i < nTopClasses; ++i) {
	      cout << top_classes[i].first << " " << top_classes[i].second << " " << zero_layer << " " << zero_perc << endl;
      }

      /*
      int layer_to_save = 3;

      auto layer = overfeat::get_output(layer_to_save);
      real* layer_data = THTensor_(data)(layer);
      long
      sf = layer->stride[0],
      sy = layer->stride[1],
      sx = layer->stride[2];
      cout << layer->size[0] << " " << layer->size[1] << " " << layer->size[2] << endl;
      for (int i = 0; i < layer->size[0]; ++i){
	      for (int y = 0; y < layer->size[1]; ++y){
	        for (int x = 0; x < layer->size[2]; ++x){
            cout << layer_data[i*sf+y*sy+x*sx] << " ";
            cout << endl;
          }
        }
      }
     
      ofstream image_file;
      image_file.open("layer.npp", ios::out);
      image_file << "P2\n";
      image_file << layer->size[1] << " " << layer->size[2] << endl;
      image_file << "10\n";
      for (int x = 0; x < layer->size[1]; ++x){
	      for (int y = 0; y < layer->size[2]; ++y){
            image_file << layer_data[y*sy+x*sx] << " ";
        }
        image_file << endl;
      }
      image_file.close();
      */

      /*
      cout << "num layers: " << overfeat::get_n_layers() << endl;
      for(int i = 0; i < overfeat::get_n_layers(); i++) {
        auto layer = overfeat::get_output(i);
        cout << "layer " << i << " dimension: " << layer->nDimension << endl;
      }
      */

    } 
    else {// if nTopClasses < 0, we output the features

      // extract features
      THTensor* output = overfeat::fprop(input_raw, zero_layer, zero_perc, mask_layer);

      // print output
      THTensor* features = overfeat::get_output(feature_layer);
      real* data = THTensor_(data)(features);
      long
      sf = features->stride[0],
      sy = features->stride[1],
      sx = features->stride[2];
      cout << features->size[0] << " " << features->size[1] << " " << features->size[2] << endl;
      for (int i = 0; i < features->size[0]; ++i){
	      for (int y = 0; y < features->size[1]; ++y){
	        for (int x = 0; x < features->size[2]; ++x){
            cout << data[i*sf+y*sy+x*sx] << " ";
            cout << endl;
          }
        }
      }

    }
  }
  
  THTensor_(free)(probas);
  THTensor_(free)(input);
  THTensor_(free)(input_raw);
  overfeat::free();
  return 0;
}
