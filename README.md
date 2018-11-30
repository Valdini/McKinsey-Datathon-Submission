# McKinsey-Datathon-Submission

Link to Google Colab: https://colab.research.google.com/drive/1XuBeNwJUwrZJHNTRnYncUOnmgCLBEHBe

- Part A - Insights
- Part B - Files Import and Data Preparation
- Part C - Exploratory Data Analysis (EDA)
- Part D - Data Cleaning & Preprocessing
- Part E - Training and Evaluation with XGBClassifier


Part A - Insights 

Brief approach: I started with an initial hypothesis on which features might help the UK government understand which actions to take in order to mitigate road accidents. After I imported the CSV files to Google Colab and shuffled the datasets, I performed Exploratory Data Analysis (EDA), including shapes and data types, check for imbalanced classes, correlation matrix to check 
for redundant features and scanning for missing values. I then cleaned and preprocessed the data by removing redundant values, dropping columns, imputing missing values based on their mode i.a. I encoded and scaled the features, divided into training and testing set and used the popular XGBClassifier as classification algorithm for the binary road accident fatality classification problem.

I applied above feature engineering and data preprocessing ideas based on previous experience and observations made in the dataset (for instance: "How many dimensions has feature X?", "How do features Y and Z correlate?" etc.).

I chose my final model (the XGBClassifier) based on previous experience and advise from multiple experts and experience data scientists. I wanted to perform hyperparameter optimization based on GridSearchCV on the XGBClassifier, but lacked time to complete this final step.
Optimizing hyperparameters based on GridSearchCV thus would be my top approach that I would try to improve my score.

Takeaways: I should have initially spent more time on intuition (for instance: "Which variables are even under control of the UK government?") and less on data exploration. Good intuition vastly decreases the time needed to perform exploration and subsequently preprocessing.


Part B - Files Import and Data Preparation

Google Colab was used as a private, cloud-based, open-source solution that handles all dependencies and simplifies the process
NOTE: CODE IS OPTIMIZED FOR SIMPLE USE ON GOOGLE COLAB. SO PLEASE RUN THE CODE IN YOUR BROWSER (NO DEPENDENCIES NEEDED, JUST LOGIN WITH GOOGLE ACCOUNT): LINK: https://colab.research.google.com/drive/1XuBeNwJUwrZJHNTRnYncUOnmgCLBEHBe
In top bar of Google Colab, select "Runtime" > "Change runtime type" > GPU to run much faster (especially important for one-hot encoding which takes long due to high dimensions for some features)

Part C - Exploratory Data Analysis (EDA)

About Data
- Size Vehicles: 451'397 (so more cars than accidents, makes sense), Test: 130k, Train: 136k (about the same)
- Key Question for Feature Selection: WHICH VARIABLES CAN UK GOVT EVEN INFLUENCE?
- Groups of Variables
  Accident: ID, Vehicles, Casualties, 
  Location-specific (UK govt influencable): Police Force, Longitude, Latitude, Local Authority District, Local Authority ONS, Urban or Rural Area, Speed Limit
  Permanent Road Conditions (UK govt influencable): Junction Control, Junction Category Road #1, Junction Category Road #2, # for 1st Road, # for 2nd Road, Pedestrian Control, Pedestrian Facilities
  Accident-specific: Police Officer Attendance (UK), Date, Time, Weather Conditions, Lightning (UK), Type of Road (UK)
  Vehicles - Accident: Manoeuvre, Skidding, Object in Carriage, Object of Carriage, Vehicle Location, Junction Location, Vehicle Leaving Carriageway, 1st Point of Impact
  Vehicles - Personal: ID, Journey Purpose, Sex, Age, Age Group, IMD Index (?), Driver Area, Type of Vehicle, Articulation, Left Hand  Drive, Engine Capacity, Vehicle Fuel Type, Age of Vehicle, Vehicle Reference
  Temporary Road Conditions: Surface, Special Conditions, Carriageway Hazards
  Target: Severity of accident

Potential Selection of Algorithms to be tested
- Linear Regression, Logistic Regression (seems to perform well on these tasks)
- Decision Tree
- SVM (SVC, allows for special handling of imbalanced classes)
- Naive Bayes
- KNN
- KMC
- Random Forest
- Gradient Boosting
- XGBoost Classifier (very popular, performs well, Scikit-Learn, p.33)
- SGD Classifier (recommended by Scikit-Learn)


Part C & D - Summary of Exploratory Data Analysis (EDA, Part C) as a To-do List for Data Cleaning & Preprocessing (Part D)

- Columns to be dropped (because of NaNs, curse of dimensionality etc.):
  Train_data: weather_conditions, time, date, location_easting_osgr, location_northing_osgr, lsoa_of_accident_location, police_force, local_authority_district, local_authority_highway, junction_control, 2nd_road_class
  Vehicle_data: Vehicle_IMD_Decile, Engine_Capacity_(CC), Age_of_Vehicle, Driver_Home_Area_Type, 
    potentially to be dropped if not performing well: Towing_and_Articulation, Vehicle_Manoeuvre, Vehicle_Location-Restricted_Lane, Junction_Location, Skidding_and_Overturning, Hit_Object_in_Carriageway            
                                                      Vehicle_Leaving_Carriageway, Hit_Object_off_Carriageway, Was_Vehicle_Left_Hand_Drive?, Propulsion_Code, Age_of_Driver

- Columns that need missing value imputation (choosing the mode as imputation strategy):
  Train_data: lsoa_of_accident_location
  Vehicle_data: Towing_and_Articulation, Vehicle_Manoeuvre, Vehicle_Location-Restricted_Lane, Junction_Location, Skidding_and_Overturning, Hit_Object_in_Carriageway, Vehicle_Leaving_Carriageway, Hit_Object_off_Carriageway, 1st_Point_of_Impact, Was_Vehicle_Left_Hand_Drive?

- Type 'Object' columns with label and one-hot encoding:
  Train_data: road_type, junction_detail, 1st_road_class, pedestrian_crossing-human_control, pedestrian_crossing-physical_facilities, light_conditions,
              road_surface_conditions, special_conditions_at_site, carriageway_hazards
  Vehicle_data: 1st_Point_of_Impact, Sex_of_Driver (Drop 'Not known'), Journey_Purpose_of_Driver

- Some object columns are worth more advanced preprocessing
  Train_data: date (clean by removing year and day), time (here makes sense to int() and StandardScale), 
  Vehicle_data: Vehicle_Type(19, regrouping), 

- All int64 and float64 columns to be standardized/StandardScaled

- Handle imbalanced dependent variable/target: put higher emphasis on 1's than on O's

- Check for PCA (Principal Component Analysis), to determine which features cause the most variance in regards to the dependent variable

- Join Training Data with Vehicle Data on accident_id (several vehicles per accident), was no time left

- Train_test_split

- Apply Models



