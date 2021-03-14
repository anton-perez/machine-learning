from parse_line import parse_line 
import sys
sys.path.append('src')
from dataframe import DataFrame

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

df = df.select_columns(new_columns)

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

df = df.select_columns(new_columns)

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