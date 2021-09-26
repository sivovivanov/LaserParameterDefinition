#--------------------------
# FIBRE LASER FOLDER
#--------------------------

This folder contains 3 sub-folders:
	- bayesian opt code:
		Single .py file that runs mock optimisation. Bayesian Optimisation code has been set up
		the way it was set up when optimising thel laser.
		!PRINTS RESULTS FROM ACTUAL OPTIMISATION AT THE END!
		
	- best optimisation runs:
		Contains checkpoint.pkl file which contains all the best results from the actual optimisations
		!THIS .pkl FILE IS USED TO PRINT THE RESULTS!
	
	- plots:
		Contains a single plot comparing the best achieved result(orange) to the target(grey)
		
#--------------------------
# LWFA FOLDER
#--------------------------

This folder contains 3 sub-folders:
	- bayesian opt code:
		Two .py file that run mock optimisations. Bayesian Optimisation code has been set up
		the way it was set up when optimising thel laser.

		One is for a0 only optimisation (run03)
		One is for a0 and w0 optimisation (run04)
		
		!PRINTS RESULTS FROM ACTUAL OPTIMISATIONS AT THE END!
		
	- best optimisation runs:
		Contains two checkpoint.pkl files which containsall the best results from the actual optimisations
		
		One is for a0 only optimisation (run03)
		One is for a0 and w0 optimisation (run04)
		
		!THESE .pkl FILES ARE USED TO PRINT THE RESULTS!
	
	- plots:
		Contains a two folders with plots from the respective optimisations.
		run04_plotter.py in "a0 and w0" folder contains the code for the plots
		
#--------------------------
# LSTM FOLDER
#--------------------------

This folder contains 1 folder and 6 files:
	- data:
		This folder contains two .npz files that can be used to train the LSTM Model. 
		!THESE FILES CONTAIN ACTUAL LASER SIMULATIONS AND ARE A PART OF THE 158 TOTAL FILES USED FOR TRAINING!
	
	- target.npz
		File containin the Target spectrum (The one we were trying to re-produce with bayesian optimisation)
		
		!TO BE USED TO PREDICT PARAMETERS WITH THE MODEL!
	
	- fibre_laser_lstm.py / fibre_laser_lstm.ipybn
		Actual code for training the LSTM model.
		Provided as a .py and .ipybn file.
		Same code in both files.
		!NOTEBOOK FILE REQUIRES CHANGES IN THE PATHS USED, IT USES THE PATHS FROM GOOGLE COLAB!
		
	- model_save.pt
		Saved state_dict() of the PyTorch Model after training.
		
		!TO BE LOADED AND USED TO PREDICT PARAMETERS!
	
	- parameter_scaler.pkl
		MinMaxScaler from scikit-learn used to scale the y data. Has been fit and stored.
		
		!TO BE USED TO INVERSE SCALE PREDICTED PARAMETERS!
		
	- parameter_predicter.py
		Loads in model_save.pkl and parameter_scaler.pkl, creates trained model.
		
		!TO BE USED TO PREDICT PARAMETERS WITHOUT TRAINING THE MODEL!		