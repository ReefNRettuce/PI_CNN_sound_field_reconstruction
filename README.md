# PI_CNN_sound_field_reconstruction
a physics informed convolutional neural network 

Read the wiki for more information. 

This is almost a direct copy of the paper : Physics-informed convolutional neural network with bicubic spline interpolation for sound field estimation (https://arxiv.org/abs/2207.10937) 

I could not have made this work without the help of the book Numerical Recipes: the Art of Scientific Computing 3rd edition. To all the authors of this book thank you so much. 

The code is kind of a mess. To train the network use trainV3a -> analysis.py this shows the results. Make sure to configure the place where you need to pull data from. I used matlab PDE solver to generate synthetic data. 

If you find any mistakes please let me know. Look especially into the helmholtz solver. This is likely where I forsee a mistake. 

Its likely this model could be much smaller. 

I haven't tested it on anything but my validation data set.

I used an 80/20 split. There are 4096 different rooms. 

This only solves for one frequency 300 Hz 

No quantization has been used. 
