
# Import the relevant libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


# The custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std) # Assigning the scaler object to the instance variable self.scaler
        self.columns = columns # Assigning the list of column names to the instance variable self.columns
        self.mean_ = None # Initialize mean attribute to None for later calculation
        self.var_ = None # Initialize var attribute to None for later calculation

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y) # Fit the %%writefile square_module.pyscaler object to the assigned list of columns among the X columns
        self.mean_ = np.mean(X[self.columns], axis=0) # Compute the mean of the fitted columns column_wise and assign it to the instance variable self.mean
        self.var_ = np.var(X[self.columns], axis=0) # Compute the variance of the fitted columns column_wise and assign it to the instance variable self.var
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns # Specify the initial column order
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns) # Transform the selected columns of input data X in a DataFrame X_scaled
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)] # Select columns from input data X that are not in self.columns in a DataFrame X_not_scaled
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order] # Return the concatenation of the two DataFrame in a specified column order


# The abseenteeism model class we are going to use for prediction
class absenteeism_model():

    def __init__(self, model_file, scaler_file):
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file: # Read the 'model' and 'scaler' file we saved earlier
            self.reg = pickle.load(model_file) # Create an instance variable self.reg that reads and open the model file
            self.scaler = pickle.load(scaler_file) # Create an instance variable self.scaler that reads and open the scaler file
            self.data = None

    def load_and_clean_data(self, data_file): # Preprocessing a new data file
        
        df = pd.read_csv(r'C:\Users\HP\Downloads\Absenteeism_new_data.csv', delimiter=',') # Import the new data
        df = df.drop_duplicates() # Drop duplicate values
        self.df_with_predictions = df.copy() # Create a checkpoint for later use
        df.drop(['ID'], axis=1, inplace=True) # Drop the 'ID' column
        df['Absenteeism Time in Hours'] = 'NaN' # Add a new column with 'Nan' string to preserve the code we created earlier

        # Get the dummies for the reasons column in a separate DataFrame
        reasons_column = pd.get_dummies(df['Reason for Absence'], drop_first=True)

        # Split the reasons columns into four types
        reason_type_1 = reasons_column.loc[:, 1:14].max(axis=1) # grouping the first 14 columns and getting the maximum
        reason_type_2 = reasons_column.loc[:, 15:17].max(axis=1) # grouping the 15th-17th columns and getting the maximum
        reason_type_3 = reasons_column.loc[:, 18:21].max(axis=1) # grouping the 18th-21st columns and getting the maximum
        reason_type_4 = reasons_column.loc[:, 22:].max(axis=1) # grouping the 22nd-28th columns and getting the maximum

        df = df.drop(['Reason for Absence'], axis=1) # Drop the 'Reason for Absence' column to avoid multicollinearity
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1) # Merging the newly created reason columns

        # Assigning names to the four reason column types
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 
                        'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason 1', 'Reason 2', 'Reason 3', 'Reason 4']
        df.columns = column_names

        # Re-order the column names
        column_names_reordered = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age', 
                                  'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]

        # Working with the 'Date' column
        df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y') # Convert to datetime
        list_month = [date.month for date in df['Date']] # Create a list with month values retrieved from the 'Date' column
        df['Month Value'] = list_month # Insert the month value in a new column in df
        df['Day of the week'] = df['Date'].apply(lambda x: x.weekday()) # Extract the weekdays from the 'Date' column and store it a new column in df
        df = df.drop(['Date'], axis=1) # Drop the 'Date' column to avoid multicollinearity

        # Re-order the column names
        column_names_reordered_2 = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Month Value', 'Day of the week', 'Transportation Expense', 
                                    'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets', 
                                    'Absenteeism Time in Hours']
        df = df[column_names_reordered_2]

        # Get the dummies of the 'Education' column
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1}) 

        df = df.fillna(value=0) # replace the 'NaN' values
        df = df.drop(['Absenteeism Time in Hours', 'Daily Work Load Average', 'Distance to Work', 'Day of the week'], axis=1) # Drop the absenteeism time column and the irrelevant variables

        self.preprocessed_data = df.copy() # Create a checkpoint for the preprocessed data in an instance variable self.preprocessed
        self.data = self.scaler.transform(df) # Standardizing specific columns in the DataFrame

    # Function outputing the probability of a datapoint to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    # Function outputing 0 or 1 based on the model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # Predict the outputs and probabilities, then add the predictions in new columns in the DataFrame
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Predictions'] = self.reg.predict(self.data)
            return self.preprocessed_data
