# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

<img width="1088" height="465" alt="image" src="https://github.com/user-attachments/assets/c95e7b00-0ee2-40a8-b561-6aeb670eefc7" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: VARNIKA.P
### Register Number: 212223240170
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,5)
        self.fc3 = nn.Linear(5,2)
        self.fc4 = nn.Linear(2,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.fc4(x)

    return x



# Initialize the Model, Loss Function, and Optimizer

ai_world = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_world.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_world(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_world.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="223" height="543" alt="image" src="https://github.com/user-attachments/assets/79969344-119d-479e-944a-90313589ab05" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="754" height="552" alt="image" src="https://github.com/user-attachments/assets/2fca981a-05fc-44eb-b622-1a95a1ccc909" />


### New Sample Data Prediction

<img width="939" height="287" alt="image" src="https://github.com/user-attachments/assets/8a856132-366c-4d6b-9dfd-ebe9c1a8f6ed" />


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
