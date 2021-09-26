import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        W1,b1,W2,b2 = self.parameters['W1'],self.parameters['b1'],self.parameters['W2'],self.parameters['b2'];
        batch_size = x.size()[0];
        self.cache['z1']=torch.zeros(batch_size,W1.size()[0]);
        self.cache['z2']=torch.zeros(batch_size,W1.size()[0]);
        self.cache['z3']=torch.zeros(batch_size,W2.size()[0]);
        self.cache['y']=torch.zeros(batch_size,W2.size()[0]);
        z1 = self.cache['z1'];z2 = self.cache['z2'];z3 = self.cache['z3'];y = self.cache['y'];
        ReLU = torch.nn.ReLU(); 
        #use pytorch's ReLU to make the code looks less nasty
        # to implement this manually, just replace the ReLU below with max(0,X)
        for i in range(batch_size): 
            z1[i] = W1 @ x[i]  + b1;
            z2[i] = ReLU(z1[i]) if self.f_function=='relu' else (torch.sigmoid(z1[i]) if self.f_function=='sigmoid' else z1[i]);
            z3[i] = W2 @ z2[i] + b2;
            y[i] = ReLU(z3[i]) if self.g_function=='relu' else (torch.sigmoid(z3[i]) if self.g_function=='sigmoid' else z3[i]);

        self.cache['x'] = x;
        self.cache['y'] = y;
        
        return y
        # TODO: Implement the forward function
        pass
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        def ReLU_grad(z):
            output = torch.clone(z);
            output[output>0]=1;
            output[output<=0]=0;
            return output
        x,z1,z2,z3,y = self.cache['x'],self.cache['z1'],self.cache['z2'],self.cache['z3'],self.cache['y'];
        batch_size = x.shape[0];
        for i in range(batch_size):
            dydz3 = 1 if self.g_function == 'identity' else (
                torch.sigmoid(z3[i])*(1-torch.sigmoid(z3[i])) if self.g_function=='sigmoid' else (ReLU_grad(z3[i])));
            dz2dz1 = 1 if self.f_function == 'identity' else (
                torch.sigmoid(z1[i])*(1-torch.sigmoid(z1[i])) if self.f_function=='sigmoid' else (ReLU_grad(z1[i])));
            self.grads['dJdb2'] += dJdy_hat[i] * dydz3;
            self.grads['dJdW2'] += torch.reshape(dJdy_hat[i] * dydz3,(dJdy_hat[i].shape[0],1)) @ torch.reshape(z2[i],(1,z2[i].shape[0]));
            self.grads['dJdb1'] += dJdy_hat[i] * dydz3 @ self.parameters['W2'] * dz2dz1;
            self.grads['dJdW1'] += torch.reshape(dJdy_hat[i] * dydz3 @ self.parameters['W2'] * dz2dz1,(z1.shape[1],1)) @ torch.reshape(x[i],(1,x.shape[1]));
        pass

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    pass
    return float(torch.sum((y-y_hat)**2)/y.shape[0])/y.shape[1],(2*y_hat-2*y)/y.shape[0]/y.shape[1]



    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    pass
    return float(-1 * torch.sum(y*torch.log(y_hat)+(1-y)*torch.log(1-y_hat))/y.shape[0]/y.shape[1]),(y_hat-y)/(y_hat-y_hat*y_hat)/y.shape[0]/y.shape[1]
    # return loss, dJdy_hat
