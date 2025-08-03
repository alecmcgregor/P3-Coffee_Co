all: project3.exe run

project3.exe:
	g++ -std=c++14 -o project3.exe code/Linear_Regression/main.cpp code/Linear_Regression/LinearRegression.cpp

run: project3.exe
	./project3.exe "code/data_generation/generated_coffee.csv" 0.0053 10000