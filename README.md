Basic usage:

Run makein src/ 
And then ./ts

Matlab:
View the Matlab source files in the MatlabSrc folder

* Run labels.m to view each label in the volume matrix
* Run writeFile.m to generate a volume matrix and write into a file of a name of your choice. 
	* Provide the dimensions of the volume on line 17
* Run plotFile.m to view the volume matrix generated from the CUDA program.


Important points:
* Dimensions mentioned in the writeFile.m on line 17 must match with the dimensions on line 47 in tesst.cu for field_size
* Provide the right filename on line 415 for readFile
* Provide any name for the output filename on line 689 for writeFile for the 5th argument
	* Make sure the filenames match in plotFile.m 

