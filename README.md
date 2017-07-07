# Speaker Change Detection

Repository for the project done at [LEAP Lab](http://leap.ee.iisc.ac.in/), Indian Institute of Science, under the guidance of [Neeraj Sharma](http://ece.iisc.ernet.in/~neeraj_sharma/) and [Prof. Sriram Ganapathy](http://www.leap.ee.iisc.ac.in/sriram/). 
The aim was to develop a model to detect speaker change in a given wave file.

## Dataset
TIMIT Dataset was used. Three disjoint set of speakers was created, and only SX and SI sentences for each speaker were taken into consideration while creating the files.

## Directory contents

* *lists* contains different lists like which files were used for training, testing and validation. It also has scripts to generate the lists. 
* *models* contains different classifiers, like DNN and CNN for different features.
* *scripts* contains data processing codes, like for generating gammatone, fbank, and other features. It also has scripts for combining the features generated according to context.
* *resources* contains plots used to visualize the features, and some generic diagrams used in the presentation.

## References

* *htkmfc.py* from [here](https://github.com/syhw/timit_tools/blob/master/src/htkmfc.py)
* Base *Gammatone* scripts from this [page](http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/)

## Contact

I can be contacted at sidm@iitk.ac.in
