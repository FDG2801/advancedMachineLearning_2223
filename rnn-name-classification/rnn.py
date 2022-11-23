#name classification
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

from utils import ALL_LETTERS,N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example

#implementing the RNN
class RNN(nn.Module):
    #nn.RNN is already in pytorch -> now is from scratch
    '''
    The RNN Name Classification Module has an input and an internal state, internally we combine them and process our combined tensor, then apply two layers and we have
    two output, a hidden one for the next input and an output one. We apply softmax because we are doing classification and then show
    '''
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN, self).__init__()

        self.hidden_size=hidden_size
        self.i2h = nn.Linear(input_size+hidden_size,hidden_size) #hidden size combined
        self.i2o = nn.Linear(input_size+hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1) #1,57 

    def forward(self, input_tensor, hidden_tensor):
        ''' 
        we return the output tensor and the hidden tensor
        '''
        combined=torch.cat((input_tensor,hidden_tensor),1)
        hidden=self.i2h(combined)
        output=self.i2o(combined)
        output=self.softmax(output)

        return output,hidden 
    
    def init_hidden(self):
        return torch.zeros(1,self.hidden_size) #return an initialized hidden state 

category_lines, all_categories=load_data() #dictionary and list of countries
n_categories = len(all_categories)
#print(n_categories) #18

n_hidden=128 #hyperparameter 
rnn=RNN(N_LETTERS, n_hidden, n_categories)

#one step
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden=rnn(input_tensor,hidden_tensor)
#print(output.size())
#print(next_hidden.size())

#each single character is an input so we need to repeat this for all letters

#whole sequence/name
input_tensor = line_to_tensor("Francesco")
hidden_tensor = rnn.init_hidden()

output, next_hidden=rnn(input_tensor[0],hidden_tensor) #we have to apply it more than once 
#print(output.size())
#print(next_hidden.size())

def category_from_output(output):
    '''
    likelihood of each character
    '''
    category_index=torch.argmax(output).item()
    return all_categories[category_index]

#print(category_from_output(output))

'''
Training phase 
'''
criterion=nn.NLLLoss() #negative log likelihood
learning_rate=0.005 #might want to play around
optimizer=torch.optim.SGD(rnn.parameters(),lr=learning_rate) #Stochastic Gradient Disc

def train(line_tensor,category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]): #length of the name
        output,hidden=rnn(line_tensor[i],hidden) #new hidden state
    loss=criterion(output,category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output,loss.item()

current_loss=0 #beginning
all_losses=[]
plot_steps,print_steps=1000,5000
n_iters=100000 
for i in range(n_iters):
    category,line,category_tensor,line_tensor=random_training_example(category_lines,all_categories)

    output,loss=train(line_tensor,category_tensor)
    current_loss+=loss

    if (i+1)%plot_steps==0: 
        all_losses.append(current_loss/plot_steps)
        current_loss=0 #edited before, neeed to be 0 again
    
    if (i+1)%print_steps==0:
        guess=category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i} {i/n_iters} {loss:.4f} {line} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n>{input_line}")
    with torch.no_grad():
        line_tensor=line_to_tensor(input_line)
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]): #length of the name
            output,hidden=rnn(line_tensor[i],hidden) #new hidden state
        guess=category_from_output(output)
        print(guess)


while True:
    sentence=input("Input:")
    if sentence=="quit":
        break
    predict(sentence)