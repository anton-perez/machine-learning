from parse_line import parse_line 
import sys
sys.path.append('src')
from dataframe import DataFrame
from linear_regressor import LinearRegressor

data_types = {
    "PassengerId": int,
    "Survived": int,
    "Pclass": int,
    "Name": str,
    "Sex": str,
    "Age": float,
    "SibSp": int,
    "Parch": int,
    "Ticket": str,
    "Fare": float,
    "Cabin": str,
    "Embarked": str
}
df = DataFrame.from_csv("kaggle/data/dataset_of_knowns.csv", data_types=data_types, parser=parse_line)
print('Testing from_csv columns...')
assert df.columns == ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
print('PASSED')

print('Testing from_csv array...')
assert df.to_array()[:5] == [[1, 0, 3, '"Braund, Mr. Owen Harris"', "male", 22.0, 1, 0, "A/5 21171", 7.25, None, "S"],
[2, 1, 1, '"Cumings, Mrs. John Bradley (Florence Briggs Thayer)"', "female", 38.0, 1, 0, "PC 17599", 71.2833, "C85", "C"],
[3, 1, 3, '"Heikkinen, Miss. Laina"', "female", 26.0, 0, 0, "STON/O2. 3101282", 7.925, None, "S"],
[4, 1, 1, '"Futrelle, Mrs. Jacques Heath (Lily May Peel)"', "female", 35.0, 1, 0, "113803", 53.1, "C123", "S"],
[5, 0, 3, '"Allen, Mr. William Henry"', "male", 35.0, 0, 0, "373450", 8.05, None, "S"]]
print('PASSED')


def get_surname(string):
  return string.split(',')[0][1:]

new_array = df.apply('Name', get_surname).to_array()
new_columns = ["PassengerId", "Survived", "Pclass", "Surname", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

df = DataFrame.from_array(new_array, new_columns)

cabin_types = []
cabin_numbers = []
for cabin in df.data_dict['Cabin']:
  if cabin == None:
    cabin_types.append(None)
    cabin_numbers.append(None)
  else:
    first_cabin = cabin.split(' ')[0]
    cabin_types.append(first_cabin[0])
    if first_cabin[1:] != '':
      cabin_numbers.append(int(first_cabin[1:]))
    else:
      cabin_numbers.append(None)

new_columns = ["PassengerId", "Survived", "Pclass", "Surname", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "CabinType", "CabinNumber", "Embarked"]
df.data_dict['CabinType'] = cabin_types
df.data_dict['CabinNumber'] = cabin_numbers

df.columns = new_columns

df = df.select(new_columns)

def represents_int(s):
  try: 
    int(s)
    return True
  except ValueError:
    return False

ticket_types = []
ticket_numbers = []
for ticket in df.data_dict['Ticket']:
  if ticket == None:
    cabin_types.append(None)
    cabin_numbers.append(None)
  else:
    if ' ' in ticket:
      reverse_ticket = ticket[::-1]
      ticket_type = reverse_ticket.split(' ', 1)[1][::-1]
      ticket_number = int(reverse_ticket.split(' ', 1)[0][::-1])

      ticket_types.append(ticket_type)
      ticket_numbers.append(ticket_number)   
    else:
      if represents_int(ticket):
        ticket_types.append(None)
        ticket_numbers.append(int(ticket))
      else:
        ticket_types.append(ticket)
        ticket_numbers.append(None)

new_columns = ["PassengerId", "Survived", "Pclass", "Surname", "Sex", "Age", "SibSp", "Parch", "TicketType", "TicketNumber", "Fare", "CabinType", "CabinNumber", "Embarked"]
df.data_dict['TicketType'] = ticket_types
df.data_dict['TicketNumber'] = ticket_numbers

df.columns = new_columns

df = df.select(new_columns)

print('Testing columns with subvariables...')
assert df.columns == ["PassengerId", "Survived", "Pclass", "Surname", "Sex", "Age", "SibSp", "Parch", "TicketType", "TicketNumber", "Fare", "CabinType", "CabinNumber", "Embarked"]
print('PASSED')

print('Testing array with subvariables...')
assert df.to_array()[:5] == [[1, 0, 3, "Braund", "male", 22.0, 1, 0, "A/5", 21171, 7.25, None, None, "S"],
[2, 1, 1, "Cumings", "female", 38.0, 1, 0, "PC", 17599, 71.2833, "C", 85, "C"],
[3, 1, 3, "Heikkinen", "female", 26.0, 0, 0, "STON/O2.", 3101282, 7.925, None, None, "S"],
[4, 1, 1, "Futrelle", "female", 35.0, 1, 0, None, 113803, 53.1, "C", 123, "S"],
[5, 0, 3, "Allen", "male", 35.0, 0, 0, None, 373450, 8.05, None, None, "S"]]
print('PASSED')

pclass_df = df.select(['Pclass', 'Survived'])
pclass_df = pclass_df.group_by('Pclass')
print('Pclass survival rate:')
print(pclass_df.aggregate('Survived', 'avg').to_array())
print('Pclass survival count:')
print(pclass_df.aggregate('Survived', 'count').to_array())

sex_df = df.select(['Sex', 'Survived'])
sex_df = sex_df.group_by('Sex')
print('Sex survival rate:')
print(sex_df.aggregate('Survived', 'avg').to_array())
print('Sex survival count:')
print(sex_df.aggregate('Survived', 'count').to_array())

sib_df = df.select(['SibSp', 'Survived'])
sib_df = sib_df.group_by('SibSp')
print('SibSp survival rate:')
print(sib_df.aggregate('Survived', 'avg').to_array())
print('SibSp survival count:')
print(sib_df.aggregate('Survived', 'count').to_array())

parch_df = df.select(['Parch', 'Survived'])
parch_df = parch_df.group_by('Parch')
print('Parch survival rate:')
print(parch_df.aggregate('Survived', 'avg').to_array())
print('Parch survival count:')
print(parch_df.aggregate('Survived', 'count').to_array())

cabin_df = df.select(['CabinType', 'Survived'])
cabin_df = cabin_df.group_by('CabinType')
print('CabinType survival rate:')
print(cabin_df.aggregate('Survived', 'avg').to_array())
print('CabinType survival count:')
print(cabin_df.aggregate('Survived', 'count').to_array())

embark_df = df.select(['Embarked', 'Survived'])
embark_df = embark_df.group_by('Embarked')
print('Embarked survival rate:')
print(embark_df.aggregate('Survived', 'avg').to_array())
print('Embarked survival count:')
print(embark_df.aggregate('Survived', 'count').to_array())

age_df = df.select(['Age', 'Survived'])
age_df = age_df.group_by_interval('Age', [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80)])
print('Age survival rate:')
print(age_df.aggregate('Survived', 'avg').to_array())
print('Age survival count:')
print(age_df.aggregate('Survived', 'count').to_array())

fare_df = df.select(['Fare', 'Survived'])
fare_df = fare_df.group_by_interval('Fare', [(0,5),(5,10),(10,20),(20,50),(50,100),(100,200),(200,999999999)])
print('Fare survival rate:')
print(fare_df.aggregate('Survived', 'avg').to_array())
print('Fare survival count:')
print(fare_df.aggregate('Survived', 'count').to_array())

new_dict = df.data_dict.copy()
new_dict["SibSp=0"] = []
new_dict["Parch=0"] = []
new_dict["CabinType=A"] = []
new_dict["CabinType=B"] = []
new_dict["CabinType=C"] = []
new_dict["CabinType=D"] = []
new_dict["CabinType=E"] = []
new_dict["CabinType=F"] = []
new_dict["CabinType=G"] = []
new_dict["CabinType=None"] = []
new_dict["CabinType=T"] = []
new_dict["Embarked=C"] = []
new_dict["Embarked=None"] = []
new_dict["Embarked=Q"] = []
new_dict["Embarked=S"] = []
for i in range(len(new_dict['Sex'])):
  if new_dict['Sex'][i] == 'male':
    new_dict['Sex'][i] = 0
  elif new_dict['Sex'][i] == 'female':
    new_dict['Sex'][i] = 1

  if new_dict['Age'][i] == None:
    new_dict['Age'][i] = 29.699

  if new_dict['SibSp'][i] == 0:
    new_dict['SibSp=0'].append(1)
  else:
    new_dict['SibSp=0'].append(0)

  if new_dict['Parch'][i] == 0:
    new_dict['Parch=0'].append(1)
  else:
    new_dict['Parch=0'].append(0)

  if new_dict['CabinType'][i] == 'A':
    new_dict['CabinType=A'].append(1)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == 'B':
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(1)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == 'C':
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(1)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == 'D':
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(1)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == 'E':
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(1)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == 'F':
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(1)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == 'G':
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(1)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == None:
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(1)
    new_dict['CabinType=T'].append(0)
  elif new_dict['CabinType'][i] == 'T':
    new_dict['CabinType=A'].append(0)
    new_dict['CabinType=B'].append(0)
    new_dict['CabinType=C'].append(0)
    new_dict['CabinType=D'].append(0)
    new_dict['CabinType=E'].append(0)
    new_dict['CabinType=F'].append(0)
    new_dict['CabinType=G'].append(0)
    new_dict['CabinType=None'].append(0)
    new_dict['CabinType=T'].append(1)

  if new_dict["Embarked"][i] == 'C':
    new_dict["Embarked=C"].append(1)
    new_dict["Embarked=None"].append(0)
    new_dict["Embarked=Q"].append(0)
    new_dict["Embarked=S"].append(0)
  elif new_dict["Embarked"][i] == None:
    new_dict["Embarked=C"].append(0)
    new_dict["Embarked=None"].append(1)
    new_dict["Embarked=Q"].append(0)
    new_dict["Embarked=S"].append(0)
  elif new_dict["Embarked"][i] == 'Q':
    new_dict["Embarked=C"].append(0)
    new_dict["Embarked=None"].append(0)
    new_dict["Embarked=Q"].append(1)
    new_dict["Embarked=S"].append(0)
  elif new_dict["Embarked"][i] == 'S':
    new_dict["Embarked=C"].append(0)
    new_dict["Embarked=None"].append(0)
    new_dict["Embarked=Q"].append(0)
    new_dict["Embarked=S"].append(1)
    

del new_dict['Parch']
del new_dict['CabinType']
del new_dict['Embarked']

new_columns = ["PassengerId", "Survived", "Pclass", "Surname", "Sex", "Age", "SibSp", "SibSp=0", "Parch=0", "TicketType", "TicketNumber", "Fare", "CabinType=A", "CabinType=B", "CabinType=C", "CabinType=D", "CabinType=E", "CabinType=F", "CabinType=G", "CabinType=None", "CabinType=T", "CabinNumber", "Embarked=C", "Embarked=None", "Embarked=Q", "Embarked=S"]

new_df = DataFrame(new_dict, new_columns)
training_df = new_df.select_rows([i for i in range(501)])
testing_df = new_df.select_rows([i for i in range(501, len(new_df.to_array()))])

print('\nSex:')
sex_df = training_df.select(['Sex', 'Survived'])
test_sex_df = testing_df.select(['Sex', 'Survived'])
survival_regressor = LinearRegressor(sex_df, 'Survived')
print(survival_regressor.coefficients)   

correct_classifications = 0   
for i in range(len(training_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: sex_df.data_dict[var][i] for var in sex_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == sex_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('train accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

correct_classifications = 0   
for i in range(len(testing_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: test_sex_df.data_dict[var][i] for var in test_sex_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == test_sex_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('test accuracy:',correct_classifications/len(training_df.data_dict['Sex']))


print('\nSex and Pclass:')
pclass_df = training_df.select(['Sex', 'Pclass', 'Survived'])
test_pclass_df = testing_df.select(['Sex', 'Pclass', 'Survived'])
survival_regressor = LinearRegressor(pclass_df, 'Survived')
print(survival_regressor.coefficients)   

correct_classifications = 0   
for i in range(len(training_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: pclass_df.data_dict[var][i] for var in pclass_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == pclass_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('train accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

correct_classifications = 0   
for i in range(len(testing_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: test_pclass_df.data_dict[var][i] for var in test_pclass_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == test_pclass_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('test accuracy:',correct_classifications/len(training_df.data_dict['Sex']))


print('\nSex, Pclass, Fare, Age, SibSp, SibSp=0, Parch=0:')
multivar_df = training_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Survived'])
test_multivar_df = testing_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Survived'])
survival_regressor = LinearRegressor(multivar_df, 'Survived')
print(survival_regressor.coefficients)   

correct_classifications = 0   
for i in range(len(training_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: multivar_df.data_dict[var][i] for var in multivar_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == multivar_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('train accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

correct_classifications = 0   
for i in range(len(testing_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: test_multivar_df.data_dict[var][i] for var in test_multivar_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == test_multivar_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('test accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

print('\nSex, Pclass, Fare, Age, SibSp, SibSp=0, Parch=0, Embarked=C, Embarked=None, Embarked=Q, Embarked=S:')
embarked_df = training_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'Survived'])
test_embarked_df = testing_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'Survived'])
survival_regressor = LinearRegressor(embarked_df, 'Survived')
print(survival_regressor.coefficients)   

correct_classifications = 0   
for i in range(len(training_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: embarked_df.data_dict[var][i] for var in embarked_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == embarked_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('train accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

correct_classifications = 0   
for i in range(len(testing_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: test_embarked_df.data_dict[var][i] for var in test_embarked_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == test_embarked_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('test accuracy:',correct_classifications/len(training_df.data_dict['Sex']))


print('\nSex, Pclass, Fare, Age, SibSp, SibSp=0, Parch=0, Embarked=C, Embarked=None, Embarked=Q, Embarked=S, CabinType=A, CabinType=B, CabinType=C, CabinType=D, CabinType=E, CabinType=F, CabinType=G, CabinType=None:')
cabin_df = training_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'Survived'])
test_cabin_df = testing_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'Survived'])
survival_regressor = LinearRegressor(cabin_df, 'Survived')
print(survival_regressor.coefficients)   

correct_classifications = 0   
for i in range(len(training_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: cabin_df.data_dict[var][i] for var in cabin_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == cabin_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('train accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

correct_classifications = 0   
for i in range(len(testing_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: test_cabin_df.data_dict[var][i] for var in test_cabin_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == test_cabin_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('test accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

print('\nSex, Pclass, Fare, Age, SibSp, SibSp=0, Parch=0, Embarked=C, Embarked=None, Embarked=Q, Embarked=S, CabinType=A, CabinType=B, CabinType=C, CabinType=D, CabinType=E, CabinType=F, CabinType=G, CabinType=None, CabinType=T:')
cabinT_df = training_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T','Survived'])
test_cabinT_df = testing_df.select(['Sex', 'Pclass', 'Fare','Age', 'SibSp=0','Parch=0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T','Survived'])
survival_regressor = LinearRegressor(embarked_df, 'Survived')
print(survival_regressor.coefficients)   

correct_classifications = 0   
for i in range(len(training_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: cabinT_df.data_dict[var][i] for var in cabinT_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == cabinT_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('train accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

correct_classifications = 0   
for i in range(len(testing_df.data_dict['Sex'])):
  prediction = round(survival_regressor.predict({var: test_cabinT_df.data_dict[var][i] for var in test_cabinT_df.columns[:-1]}))
  if prediction > 1:
    prediction = 1
  if prediction == test_cabinT_df.data_dict['Survived'][i]:
    correct_classifications+=1

print('test accuracy:',correct_classifications/len(training_df.data_dict['Sex']))

