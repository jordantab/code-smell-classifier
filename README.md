# code-smell-classifier
18668 Data Science for Software Engineering - Assignment 1 Code Smell Classifier

Assignment 1: Code Smell Classifier
 

The goal of the first assignment is to assess your ability in
training machine learning (ML) models,
interpret their results in order
to identify the bias in the dataset.

You need to tackle the following three tasks. Each task is built on the top of the previous one. Completing all the tasks does not give you 100% of the grade – some penalties may be applied! However, you need to complete all tasks to get 100%. Everything will be evaluated in this assignment, including the process to train the models, the quality of the scripts, and the discussions in the report.

 
Task 1: Training different models (4 pts)

In the first task, you need to create a python script (or scripts) to train at least 1 machine learning model. You can use any ML algorithm to train your model, for example:
Decision Tree
Random Forest
Naïve Bayes
SVM (kernels: Linear, Polynomial, RBF, Sigmoid)
You can choose another ML algorithm

Notice that you DON'T need to train the SVC with the four specified kernels if you decide to choose it as an algorithm.

During the training, you MUST split the data into training and test sets. You are allowed to use cross-validation (for comparison purpose), feature selection, grid search, ensemble learning, etc. However, the main decisions that you make should be explained in the final report, including their rationale and how they have affected your results (vide Task 3).
 
You need to train the models to classify four types of code smells: God Class, Data Class, Feature Envy, and Long Method. Each smell has its own file. You can download the files from here. You’ll find "arff files" for each smell in the folder. You do not need to merge these four files to create a single dataset. In reality, merging them is the right strategy since it will lead to a more realistic dataset, as discussed during the class. However, you do not need to do it in this assignment for the sake of simplicity but you can if you want.

 
Deliverable: you should provide the Python scripts used to train the models and also the four datasets. In addition, you should write in your report how we can execute your scripts. You will only receive 100% of the score for this task if we can execute your scripts. More details are provided in Section 4 – Deliverables.

Retrain: 

 
Task 2: Launch the application (2 pts)

Now that you have trained your models, you should wrap the scripts as a single application. Since the goal is to make it easier for the user to use your script from task 1 (i.e., allow them to train and see the results), usability is an important non-functional requirement. Therefore, you need to create another python script to run/manage your application. The script should print in the terminal the accuracy and F1 Score for each model the user decides to train. You can create a graphical user interface if you don't want to use the terminal.

Requirements: Regardless how the users will interact with the application, it should allow them:

List trained models specifying the ML algorithms: the application should allow the users to display all the trained models (including the SVM kernels) available. Hence, they can know the models available and call them individually (see the next requirement).

Show results for the classification metrics: the application should allow the users to see the classification metrics (accuracy and the F1-score) for a selected model. It should allow the user to select the model and choose one or more code smell(s), then displaying the metrics' values for the test set based on the selection. 

Feel free to decide how the user will interact with the application. For example, suppose your application is running in the terminal, and the user wants to see the classification metrics for the Decision Tree model when it was trained with the Feature Envy and God Class datasets. You can provide the following command:
>> run decision_tree feature_envy god_class

As output, the command should return the accuracy and the F1- score achieved in the test set for both provided smells. The output should be something similar to:
(output)      	Smell           	   	|      	Accuracy     	|      	F1-score
(output)      	Feature Envy   	|      	90%             	|      	90%
(output)      	God Class    	|      	90%             	|      	90%

Compare test and training sets: The application should allow the user to see the metrics comparison between the test set and training set for one smell per time. This will allow users to see how much the model's performance deviates from the training set.

For example, you can provide the following command:
>> compare decision_tree feature_envy
(output)      	Feature Envy  	|      	Accuracy     	|      	F1-score
(output)      	Training set 	|      	94%             	|      	96%
(output)      	Test set        	  	|      	90%             	|      	84%
 
 
Re-train models: The application should allow the users to re-train the models using the scripts from task 1. For example, to improve usability, you can train the model.

Notice that some models take a long time to be trained. Therefore, there is a trade-off between usability and performance. Thus, you need to make decisions to better meet certain non-functional requirements such as usability, performance and robustness. In summary, your application should be intuitive, prevent users from making mistakes, and NOT CRASH. For example, if the user types the command run decision_tree feature_envy divergent_change. Your application SHOULD NOT crash, instead, it can return a message informing an error: there is no file associated with the Divergent Change smell. I suggested you create a simple menu to let users select the available features/options. We will test whether your application crashes or not.

Feel free to make some decisions to meet other non-functional requirements. For example, you can use multithreading to train the models in parallel and make them available dynamically; thus, the user does not need to wait for all models to complete, improving usability. You can also save the results in a file, so the users do not have to train the model everytime they launch the application. These extra features are optional and if well justified in the report can guarantee you extra points.

Make sure you are following the style guide for Python code.
  
Deliverable: you should provide the script to execute your application. 
