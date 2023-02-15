Here are some notes regarding this folder and possible areas for refactoring the code

In `footandball.py`:

- [ ] Tidy up the softmax code once its verified that it works during training (as well as overfitting on a single image)
- [ ] Add tensortype annotations (or comment the shape) of tensors (particularly the ball_feature_maps) throughout the code
- [ ] Experiment between int32 and int64 where noted
- [ ] I might want to remove the raw ball feature maps from the detection output; or perhaps to just add add a flag for when
I want this option 

Notes on `footandball.py`:

- Note that the shape of _ball_feature_map_ changes between training and detect mode
  - In training mode, the shape is (batch_size, height, width, num_channels)
  - In detect mode, the shape is (batch_size, num_channels, height, width)