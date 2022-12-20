# Tools for preprocessing the hdf5 files before use with the networks

## delete_samples.py

Only keeps the first 64 samples of the time domain impulses

  * delete_sample will by default do 4 point sinewave fade in, 8 point cosine wav fade out
  * "d" has been multiplied
  * to delete pre-delay and truncate all for left and right ear do:
    
    ```bash
	python delete_samples.py -t trunc_64 -n 64 cipic all -f
    ```
