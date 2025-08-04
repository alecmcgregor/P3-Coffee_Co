<H1> COP3530 - Project 3 </h1>
These are the instructions for how to run our code.

<h2> Methods </h2>
1. Navigate to the code folder in the repository and run the command: mingw32-make<br>
2. This will automatically build the project3.exe and run both the Multivariate Linear Regression Model and the Decision Tree Regression Model.<br>
3. If you would like to run either of the models again or individually you can run them with these 2 commands:<br>
	a) ./project3.exe "Decision_Tree" "data_generation/generated_coffee.csv" 12 20<br>
 		i) The first value after the location of the data is for the max depth, and the second number is for the minimum sample split<br>
 	b) ./project3.exe "Linear_Regression" "data_generation/generated_coffee.csv" 0.0053 10000<br>
  		i) i) The first value after the location of the data is for the learning rate, and the second number is for the number of iterations<br>
4. Lastly, in order to generate the visualizations of the data run the command: python visualize_results.py<br>
