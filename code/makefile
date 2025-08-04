all: project3.exe run visualize

project3.exe:
	g++ -std=c++14 -ILinear_Regression -IDecision_Tree -o project3.exe main.cpp Linear_Regression/LinearRegression.cpp Decision_Tree/DecisionTree.cpp

run: project3.exe
	./project3.exe "Decision_Tree" "data_generation/generated_coffee.csv" 12 20
	./project3.exe "Linear_Regression" "data_generation/generated_coffee.csv" 0.0053 10000

visualize:
	python visualize_results.py
	