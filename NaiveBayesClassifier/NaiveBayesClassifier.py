import numpy as np
import pandas as pd



# Database
employee = {
    "Department": ["sales","sales","sales","systems","systems","systems","systems","marketing","marketing", "secretary","secretary"],
    "Age": ["31...35","26...30","31...35","21...25","31...35","26...30","41...45","36...40","31...35","46...50","26...30"],
    "Salary": ["46K-50K","26K-30K","31K-35K","46K-50K","66K-70K","46K-50K","66K-70K","46K-50K","41K-45K","36K-40K","26K-30K"],
    "Status": ["senior","junior","junior","junior","senior","junior","senior","senior","junior","senior","junior"],
}


data = pd.DataFrame(employee)
print("Dataset")
print(data)
print("")

# Train Naive Bayes Model
def Naive_Bayes(data):
    tables=['Department','Age','Salary']
    #Calculation Prior Probality for senior and jurior in data using numpy 
    y_unique = data['Status'].unique()
    

    prior_probability = np.zeros(len(data['Status'].unique()))
    

    for i in range(0,len(y_unique)):
        prior_probability[i]=(sum(data['Status']==y_unique[i])+1)/(len(data['Status'])+2)
    
    
    print("Prior probability after Laplac")
    print(y_unique[0],prior_probability[0])
    print(y_unique[1],prior_probability[1])
    #Conditional Probality

    #Dictionary for tables
    conditional_probability = {}

    

    #Frequency Tables
    print("")
    print("Frequency Tables with Laplac Smoothing for each class")
    for i in range(len(tables)):
        x_unique = list(set(data[tables[i]]))
        
        x_conditional_probability = np.zeros((len(data.Status.unique()),len(set(data[tables[i]]))))
        

        # adding 1 on numinator and number of entries on denominator for laplac smoothing(for removing zero probability)
        for j in range(0,len(y_unique)):
            for k in range(0,len(x_unique)):
                x_conditional_probability[j,k]=(data.loc[(data[tables[i]]==x_unique[k])&(data['Status']==y_unique[j]),].shape[0]+1)/(sum(data['Status']==y_unique[j])+len(x_unique))

        x_conditional_probability = pd.DataFrame(x_conditional_probability,columns=x_unique,index=y_unique) 
        conditional_probability[tables[i]] = x_conditional_probability 
        print("")
        print(tables[i])
        print(x_conditional_probability)

    return prior_probability,conditional_probability

prior_probability,conditional_probability=Naive_Bayes(data)           


#Prediction Function
def prediction(X):
    department=X[0]
    age=X[1]
    salary=X[2]


    #for senior
    p0=prior_probability[0]*conditional_probability['Department'][department]["senior"]*conditional_probability['Age'][age]["senior"]*conditional_probability['Salary'][salary]["senior"]
    #for junior
    p1=prior_probability[1]*conditional_probability['Department'][department]["junior"]*conditional_probability['Age'][age]["junior"]*conditional_probability['Salary'][salary]["junior"]
    
    if p0>p1:
        pred="Senior"
    else:
        pred="Junior"
    
    return pred

print("")
print("Prediction For : Marketing , 31…35, 31K-35K ")
print(prediction(['marketing','31...35','31K-35K']))
print("")
print("Prediction For : Sales, 31…35, 66K-70K  ")
print(prediction(['sales','31...35','66K-70K']))
