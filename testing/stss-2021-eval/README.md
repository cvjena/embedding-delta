Short usage guide:

* The template contains many utility packages to enable you to use it for development
* You may install any python package to the virtualenv at `/pip` by listing them in the `requirements.txt`
* You may install any ubuntu package (line 12)
* You may change the base image **at your own risk**. Please contact us to make sure that other base images will run fine. 
* The UID is fixed to the UID that the script will run with in our cluster to ensure that there are no permission
  errors. Please do not change that UID.
* Add all your code to the directory `/code` in the Docker image
    * There needs to be a python file called `predict.py` that takes no arguments, reads all files from `/data/test` and
      runs your inference on these files
    * Your code needs to write its predictions into the folder `/predictions`, creating one output file per input file
      with the same name. The output format needs to match the json input format.
* Suggestion: Mount the training data to `data/train` (not required)
* We will mount the test data to the folder `/data/test` and run your `predict.py` on these files
* We will evaluate your predictions using the
  script [eval.py](https://gitlab2.informatik.uni-wuerzburg.de/kallimachos/stss-2021-eval/-/blob/master/eval.py)
