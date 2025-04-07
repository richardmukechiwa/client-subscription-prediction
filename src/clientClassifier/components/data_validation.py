#creating components
import os
from clientClassifier import logger
from clientClassifier.entity.config_entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
    
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None
            
            data= pd.read_csv(self.config.unzip_dir)
            all_columns = list(data.columns)
            
            all_schema = self.config.all_schema.keys()
            
            for col in all_columns:
                if col in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as  f:
                        f.write(f"Column {col} not in schema")
                        
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as  f:
                        f.write(f"Validation status: {validation_status}")
                        
            return validation_status
        
        except Exception as e:
            logger.exception(e)
            raise e  

    def validate_data_types(self)-> bool:
        try:
            data_type_status = None
            
            data= pd.read_csv(self.config.unzip_dir)
            
            #checking the data types
            data['y'] = data['y'].map({'yes': 1, 'no': 0})
            #print("dtypes here", data.dtypes)
            
            datatype = dict(data.dtypes)
            #print("datatype here", datatype)
            
            
            all_schema_val = dict(self.config.all_schema)
            
            #print("all schema here", all_schema_val)
            
            for col, dat in datatype.items():
                if col not in all_schema_val or str(dat) != all_schema_val[col]:
                    data_type_status = False
                    with open(self.config.STATUS_FILE, 'w') as  f:
                        f.write(f"Data type different from the one  in the schema")
                else:
                    data_type_status = True
                    with open(self.config.STATUS_FILE, 'w') as  f:
                        f.write(f"Validation status: {data_type_status}")
                        
            return data_type_status
        
        except Exception as e:
            logger.exception(e)
            raise e  
   