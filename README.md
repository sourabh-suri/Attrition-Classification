<h1>Attrition Classification</h1> 


Implement any classifier using library functions to predict whether an employee will leave
the company or not.
Data: Test and Train data are given with 33 and 34 columns respectively and 1028 rows.
The column vector consists of:
Age, Attrition, BusinessTravel, DailyRate, Department,
DistanceFromHome, Education, EducationField, EmployeeCount,
EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate,
JobInvolvement, JobLevel, JobRole, JobSatisfaction,
MaritalStatus, MonthlyIncome, MonthlyRate,
NumCompaniesWorked, OverTime, PercentSalaryHike,
PerformanceRating, RelationshipSatisfaction, StockOptionLevel,
TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
YearsWithCurrManager, ID


<h2> Data Processing:</h2>


• Encoding


The first step taken was to encode the data such that we are left with numerical values only
for the categorical data. Label Encoding converts categorical labels into numerical values.
For the categorical data which have more than two unique values are encoded using One Hot
Encoding. Features like, BusinessTravel, Department, OverTime, etc are converted based on
their unique column values. One hot encoding is used to avoid any bias with large values
assigned to data having more than two unique values.


• Scaling


The classification algorithms works best for similarly scaled features. Thus feautre scaling is
apllied using MinMaxScaler to range values between 0 to 1.


• Removing Target and redundant features


The target feature : Attrition is removed from train data for model training.
Also the features like- EmployeeNumber and ID which are unique for every row are
removed as they will not help in any training. The feature called EmployeeCount which has
same value all over is also removed.


Lastly the feautures like, MonthlyRate, HourlyRate and DailyRate seems correlated and
removed except one for reducing irrelevant features. Thus reducing efforts on model
training.


After above data processing, the columns for test and train data are,
Age, DailyRate, DistanceFromHome, Education,
EnvironmentSatisfaction, Gender, JobInvolvement, JobLevel,
JobSatisfaction, MonthlyIncome, NumCompaniesWorked,
OverTime, PercentSalaryHike, PerformanceRating,
RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany,
YearsInCurrentRole, YearsSinceLastPromotion,
YearsWithCurrManager, BusinessTravel_Travel_Frequently,
BusinessTravel_Travel_Rarely, Department_Research & Development,
Department_Sales, EducationField_Life Sciences,
EducationField_Marketing, EducationField_Medical,
EducationField_Other, EducationField_Technical Degree,
JobRole_Human Resources, JobRole_Laboratory Technician,
JobRole_Manager, JobRole_Manufacturing Director,
JobRole_Research Director, JobRole_Research Scientist,
JobRole_Sales Executive, JobRole_Sales Representative,
MaritalStatus_Married, MaritalStatus_Single

<h1>Train-Test data split & Cross validation</h1> 


Using train_test_split helper function from scikit-learn, one can evaluate hyper paramaters for their
classifier. However, there is a risk of overfitting on the test set because the hyper-parameters can be
tuned for optimal values. By spliting the available data into two sets, the learning model can be
deprived of the sufficient number of training samples. Also, a particluar choice of training and
validation data may increase the accuracy of overall result but have poor predictability for the new
data.


</h2>Cross validation</h2> 

It helps to solve the above problem where training samples are split into k sets
(k-folds) for validation. An average of the k folds can be taken as good measure of accuracy for that
model.

