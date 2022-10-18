METHOD
	My autofit method works as follows: After recieving the x, y, gamma, and maximum degree
inputs, the method looped through each degree and called the regularized least squares
function to calculate the optimal weights for the given degree and output the predicted target
variable. Then, the rmse and variance [rss/(n-m-1)] were both calculated and added into a
list. After the loop, each error list was min/max normalized. The rmse was weighted by 0.1 and
the two errors were added together. So, my heuristic error function is as follows:
		heuristic error = 0.1*minmaxnorm(rmse) + minmaxnorm(var)
As the order increases, a decrease of rss becomes less significant (e.g a bigger decrease 
in bias is necessary to lower variance metric), which allows the heurisitic error to account
for added degrees that dont vastly improve the model. Further, we implement RMSE to 
re-emphasize minimizing bias (the variance function already does, but I found that adding a 
small bias parameter improved the order accuracy on the labelled data significantly). Therefore, 
the heuristic function looks to minimize both bias AND variance in the model when choosing the 
optimal order. Finally, np.argmin was called to find the degree (index+1) that minimized 
heuristic error. Optimal order, as well as weights and error scores were printed to the console.	

Note: My autofit method is stable and accurate up until the degree approaches the size of the data.
This is because variance approaches inf @ size(data)-1 (messes with normalization). Autofit order 
accuracy also depends on the regularization parameter (see next section).


REGULARIZATION
	To choose the ideal regularization parameter, I tested with my autofit method and
looked specifically at the higher order labelled data first. The ballpark optimal consistent 
regularization parameter was 0.1. This allowed for the minimization of overfitting and large
coefficients while also balancing accuracy of model order across all LevelOne data. This yielded
the most accurate order for B, C, D, E.
	For the linear case (A), it is ideal to choose larger regularization parameter to prevent 
overfitting. Here, a value of 0.3 will choose the correct order while still minimizing
RMSE.


MODEL STABILITY
	My autofit method will consistently return the same optimal order and weights when run
multiple times on the same input parameters and data. Further, when randomly subsetting the 
data (using the random package) at 80%, the autofit method will match the optimal order of 
the full dataset at around a 75% rate. Therefore, the model is stable across multiple conditions.


RESULTS
	LevelOne Data: For the LevelOne data, my autofit method correctly predicted the order 
of best fit polynomials A,B, and C with the majority of parameters being accurate to the 
hundredth place, and the rest to the tens place. For D and E, the autofit method fell 
within a degree of the correct order. 

	SecondSet Data: For the unknown data, at a chosen maximum degree of 48 and gamma of
0.1, my autofit method predicted as follows (errors also printed but not shown):

Dataset X: 
Autofit order chosen: 6
Weights found using RLS: [ 0.33201678 -0.24121973  1.09084316  0.51305545  0.15522779  
 0.99940346 -0.26242478]

Dataset Y:
Autofit order chosen: 9
Weights found using RLS: [ 1.96730877e+00  4.52594121e-01 -4.96774864e-01  1.09091012e-01
 3.70160233e-01  1.80970447e-04  1.06110039e+00 -8.52405836e-02
 1.43334557e+00 -1.52407768e-01]

Dataset Z: 
Autofit order chosen: 4
Weights found using RLS: [ 1.38205614  0.52509343  0.0863094  -0.81164356  0.02750078]

The order chosen for each of the three datasets very much makes sense, as shown by setting
the showPlot arg to true to view the best fit line superimposed on the data. Additionally,
both RMSE and Variance are quite low.


COLLABORATION
	I worked solo on this project!


NOTES
	The commandline arguments are as follows:
	
	Required Args
		(-m, --max) - Max degree, int (required)
		(-g, --gamma) - Gamma (default 0), float
		(-p, --trainPath) - Trainpath to dataset, string (required)
		(-o, --modelOutput) - Model file output path, string (don't include filename! ex: .\folder\)
		(-a, --autofit) - Autofit flag, no value
		(-i, --info) - Info flag, no value
		
	Additional Args
		(-s, --showPlot) - show pyplot of chosen (or autofit) polynomial, no value
