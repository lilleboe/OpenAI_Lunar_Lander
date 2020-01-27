# OpenAI Lunar Lander Machine Learning Project

#### Files/Directories
* capstone.py
  - Main Python project code. This code along with the two weight files listed below should all
be in the same directory.
* graph_utils.py
  - Set of graphing functions to output metric visualizations. This should be place in the same
directory as the capstone.py file.
* benchmark_dqn.h5
  - Weights file for optimized benchmark NN.
* solution_dqn.h5
  - Weights file for optimized solution NN.
* saved_models
  - Required directory to store periodic weight files. This directory can be initially empty.
* Images
  - Directory that holds all the generated images and screenshots used in the report. This isn’t
a required directory, but it was added for further reference.
* project_report.pdf
  - The capstone project’s report.
* proposal.pdf
  - The capstone proposal document.
  
#### Required Packages
* Python 3.6.1
  - The project was created with Anaconda 3
* Keras 2.1.2
* TensorFlow 1.12.0
* Numpy 1.14.0
* Matplotlib 2.1.2
* gym.spaces

#### Instructions
Call capstone.py from a command line:

`> Python captone.py`

In its current default settings, it will run the 100 tests against both the benchmark and solution trained
DQNs by pulling in the benchmark_dqn.h5 and solution_dqn.h5 files and render the images. If you want to
adjust from the default you can adjust a few variables to run various options. At the bottom of the file you
will see these variables and their current settings:

```
 run_benchmark_multiple_tests = False
 run_solution_multiple_tests = False
 run_benchmark_test = False
 run_solution_test = False
 run_benchmark_100_test = True
 run_solution_100_test = True
```

run_benchmark_multiple_tests & run_solution_multiple_tests will run various tests on parameter settings (This takes about 24-48 hours to run). run_benchmark_test & run_solution_test will run a training session on the parameter settings that I found to be the most optimal for both the benchmark and solution. run_benchmark_100_test & run_solution_100_test will load the ...dqn.h5 weights files and run 100 tests against the trained NN. This last set of variables is currently set to True; however, you can change the other ones, but
they will take a long time to complete. 
