import math

class NaiveBayes:
  def __init__(self, data_dict):
    self.data = data_dict
    self.vars = data_dict.keys() 

  def P(self, var): 
    return sum(self.data[var])/len(self.data[var])

  def not_P(self,var):
    return 1-self.P(var)

  def P_given(self, var1, var2, negation):
    if negation[1]:
      total_list = [i for i in range(len(self.data[var2])) if self.data[var2][i]]
    else:
      total_list = [i for i in range(len(self.data[var2])) if not self.data[var2][i]]
      
    if negation[0]:
      subset = [i for i in total_list if self.data[var1][i]]
    else:
      subset = [i for i in total_list if not self.data[var1][i]]
      
    return len(subset)/len(total_list)
  
  
  def classify(self, class_type, input_vars):
    class_probs = [self.not_P(class_type), self.P(class_type)]
    init_max_prob = class_probs.index(max(class_probs))
    for var in input_vars:
      for i in range(2):
        class_probs[i] *= self.P_given(var, class_type, [input_vars[var], i])

    if len(set(class_probs)) == 1:
      return init_max_prob

    return class_probs.index(max(class_probs))

scam_dict = {
  'scam':[0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
  'errors':[0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
  'links':[0, 1, 1, 0, 1, 1, 0, 1, 0, 1]}

NB = NaiveBayes(scam_dict)

input_vars = {'errors':0, 'links':0}
print(NB.classify('scam', input_vars))
input_vars = {'errors':1, 'links':1}
print(NB.classify('scam', input_vars))
input_vars = {'errors':1, 'links':0}
print(NB.classify('scam', input_vars))
input_vars = {'errors':0, 'links':1}
print(NB.classify('scam', input_vars))