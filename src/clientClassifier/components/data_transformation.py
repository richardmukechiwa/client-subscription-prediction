#creating the components
import os
from clientClassifier import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clientClassifier.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        logger.info(f"Data transformation started")
        
        # read the data
        df = pd.read_csv(self.config.data_path)
        
         # showing the first 5 rows of the data
        logger.info(f"Data sample: {df.head()}")
        
        
        print()
        print('<' * 70)
        print()
  
        
        # explain what each column represents 
        logger.info(f"Data columns description:")   
        logger.info(f"1. age: Age of the client")
        logger.info(f"2. job: Type of job (e.g., admin, technician, etc.)")
        logger.info(f"3. marital: Marital status (e.g., single, married, divorced)")        
        logger.info(f"4. education: Level of education (e.g., primary, secondary, tertiary)")   
        logger.info(f"5. default: Whether the client has credit in default (yes/no)")   
        logger.info(f"6. balance: Average yearly balance in euros")     
        logger.info(f"7. housing: Whether the client has a housing loan (yes/no)")      
        logger.info(f"8. loan: Whether the client has a personal loan (yes/no)")    
        logger.info(f"9. contact: Type of communication used to contact the client (e.g., cellular, telephone)")        
        logger.info(f"10. day: Last contact day of the month (1-31)")           
        logger.info(f"11. month: Last contact month of year")               
        logger.info(f"12. duration: Duration of the last contact in seconds")                       
        logger.info(f"13. campaign: Number of contacts performed during this campaign for this client")                         
        logger.info(f"14. pdays: Number of days since the client was last contacted from a previous campaign (999 means client was not previously contacted)")  
        logger.info(f"15. previous: Number of contacts performed before this campaign for this client") 
        logger.info(f"16. poutcome: Outcome of the previous marketing campaign (e.g., success, failure)")
        logger.info(f"17. y: Target variable (whether the client subscribed to a term deposit or not)")
               
        
        print()
        print('<' * 70)
        print()
        
         # check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found in the data: {missing_values}")
        else:
            logger.info(f"No missing values found in the data") 
            
        # check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Duplicates found in the data: {duplicates}")
        else:
            logger.info(f"No duplicates found in the data") 
            
        # check for outliers    
        # (this is a simple example, in practice you would use more sophisticated methods)  
        outliers = df.describe()    
        logger.info(f"Outliers in the data: {outliers}")
        
        # check for categorical variables   
        categorical_vars = df.select_dtypes(include=['object']).columns 
        
        if len(categorical_vars) > 0:
            logger.info(f"Categorical variables found in the data: {categorical_vars}")                 
        else:
            logger.info(f"No categorical variables found in the data")      
            
        # check for numerical variables
        numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns 

        if len(numerical_vars) > 0: 
            logger.info(f"Numerical variables found in the data: {numerical_vars}")         
        else:
            logger.info(f"No numerical variables found in the data")

        # check for class imbalance
        class_counts = df['y'].value_counts()      
        logger.info(f"Class distribution in the data: {class_counts}")      
        
        if class_counts.min() / class_counts.max() < 0.2:
            logger.warning(f"Class imbalance found in the data")                
        else:   
            logger.info(f"No class imbalance found in the data")   
          
        print()
        print('<' * 70)
        print()  
        
        # convert target variable to binary 
        df['y'] = df['y'].map({'yes': 1, 'no': 0}) 
        
        logger.info(f"Target variable converted to binary")
        
        print()
        print('<' * 70)
        print()
        
        #exploratory data analysis (EDA)
        df['y'].value_counts().plot(kind='bar')
        plt.title('Target Variable Distribution')   
        plt.xlabel('Target Variable')
        plt.ylabel('Count')
        plt.show()
        
        # check for feature distribution based on the target variable  y    
        for col in df.select_dtypes(include=['int64', 'float64']).columns:  
            plt.figure(figsize=(8, 4))
            sns.barplot(x='y', y=col, data=df, estimator=sum, errorbar=None, hue='y')  
            plt.title(f'Distribution of {col} by Target Variable')
            plt.xlabel('Target Variable')
            plt.ylabel(col)
            plt.show()
                
                
       # check for feature correlation     
        correlation_matrix = df.select_dtypes("number").corr()  
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.show()
               
        
        logger.info(f"Feature correlation in the data: {correlation_matrix}")           
        
        logger.info("Correlation matrix analysis complete.")
        print()
        print('<' * 70)
        print()
        # General observations
        logger.info("Most features in the dataset show low pairwise correlation, suggesting minimal multicollinearity.")
        logger.info("This is good for tree-based models which are not sensitive to multicollinearity, but may still affect linear models like logistic regression.")
        print()
        print('<' * 70)
        print()
        # Target correlation
        logger.info("'duration' has a very strong positive correlation with the target variable 'y'.")
        logger.info("However, 'duration' is a data leakage feature — it’s only known after the campaign call and directly impacts the prediction.")
        logger.warning("Including 'duration' would artificially inflate model performance. Exclude it during training to ensure real-world generalization.")
        print()
        print('<' * 70)
        print()
        logger.info("Consider removing 'duration' from the feature set to avoid data leakage.")
        logger.info("Features like 'previous' and 'pdays' also show some correlation with the target variable but")  
          
    
        print()
        print('<' * 70)
        print()
        # Next step suggestions
        logger.info("Consider conducting feature importance analysis using Random Forest or XGBoost to refine feature selection.")
        print()
        print('<' * 70)
        print()
        #drop the 'duration' column
        df.drop(columns=['duration'], inplace=True)
        
        logger.info(f"'duration' column dropped from the data")
        print()
        print('<' * 70)
        print()
        # drop the 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            logger.info(f"'Unnamed: 0' column dropped from the data")
        else:
            logger.info(f"'Unnamed: 0' column not found in the data")
            
        #drop the 'pdays' column if it exists
        if 'pdays' in df.columns:
            df.drop(columns=['pdays'], inplace=True)
            logger.info(f"'pdays' column dropped from the data")
        else:
            logger.info(f"'pdays' column not found in the data")    
            
            
        return df
           
                              
        
        # split the data into train and test sets
    def split_data(self, df):
        logger.info(f"Splitting data into train and test sets")
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        
        # save the train and test sets to csv files
        train_set.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_set.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        
        logger.info(f"Train and test sets saved to {self.config.root_dir}")
        logger.info(f"Train set shape: {train_set.shape}")
        logger.info(f"Test set shape: {test_set.shape}")
        
        
        print(f"Train set shape: {train_set.shape}")    
        print(f"Test set shape: {test_set.shape}")
        
        
        
        return train_set, test_set
    