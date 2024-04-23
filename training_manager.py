from preprocess import * 
from model import *

class PINNManager():
    def __init__(self,*args):
        super(PINNManager, self).__init__()

        self.pinn = args[0]
        self.lr= args[1] if args[1] is not None else 0.001
        self.PDE = args[2]
        self.optim = optim.Adam(self.pinn.parameters(),lr=self.lr)
    
    def predict_navier(self,x):

        out = self.pinn.forward(x)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        sig_xx = out[:, 3:4]
        sig_xy = out[:, 4:5]
        sig_yy = out[:, 5:6]
        return u, v, p, sig_xx, sig_xy, sig_yy
    
    def predict_conduction(self,x):
        T = self.pinn.forward(x)
        return T
    
    def predict_burger(self,x):
            U = self.pinn.forward(x)
            return U

    def pde_loss_navier(self,x_,rho=1,mu=0.02):
        x = x_.clone()
        x.requires_grad = True
        u, v, p, sig_xx, sig_xy, sig_yy = self.predict_navier(x)
        rho = 1
        mu = 0.02
        u_out = grad(u.sum(), x, create_graph=True)[0]
        v_out = grad(v.sum(), x, create_graph=True)[0]
        sig_xx_out = grad(sig_xx.sum(), x, create_graph=True)[0]
        sig_xy_out = grad(sig_xy.sum(), x, create_graph=True)[0]
        sig_yy_out = grad(sig_yy.sum(), x, create_graph=True)[0]
        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]
        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]
        sig_xx_x = sig_xx_out[:, 0:1]
        sig_xy_x = sig_xy_out[:, 0:1]
        sig_xy_y = sig_xy_out[:, 1:2]
        sig_yy_y = sig_yy_out[:, 1:2]
        # continuity equation
        f0 = u_x + v_y
        # navier-stokes equation
        f1 = rho * (u * u_x + v * u_y) - sig_xx_x - sig_xy_y
        f2 = rho * (u * v_x + v * v_y) - sig_xy_x - sig_yy_y
        # cauchy stress tensor
        f3 = -p + 2 * mu * u_x - sig_xx
        f4 = -p + 2 * mu * v_y - sig_yy
        f5 = mu * (u_y + v_x) - sig_xy
        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_f3 = torch.mean(torch.square(f3))
        mse_f4 = torch.mean(torch.square(f4))
        mse_f5 = torch.mean(torch.square(f5))
        mse_pde = mse_f0 + mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5
        return mse_pde

    def point_selection_navier(self,x_,percent=0.1,rho=1,mu=0.02):
        x = x_.clone()
        x.requires_grad = True
        u, v, p, sig_xx, sig_xy, sig_yy = self.predict_navier(x)
        rho = 1
        mu = 0.02
        u_out = grad(u.sum(), x, create_graph=True)[0]
        v_out = grad(v.sum(), x, create_graph=True)[0]
        sig_xx_out = grad(sig_xx.sum(), x, create_graph=True)[0]
        sig_xy_out = grad(sig_xy.sum(), x, create_graph=True)[0]
        sig_yy_out = grad(sig_yy.sum(), x, create_graph=True)[0]
        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]
        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]
        sig_xx_x = sig_xx_out[:, 0:1]
        sig_xy_x = sig_xy_out[:, 0:1]
        sig_xy_y = sig_xy_out[:, 1:2]
        sig_yy_y = sig_yy_out[:, 1:2]
        # continuity equation
        f0 = u_x + v_y
        # navier-stokes equation
        f1 = rho * (u * u_x + v * u_y) - sig_xx_x - sig_xy_y
        f2 = rho * (u * v_x + v * v_y) - sig_xy_x - sig_yy_y
        # cauchy stress tensor
        f3 = -p + 2 * mu * u_x - sig_xx
        f4 = -p + 2 * mu * v_y - sig_yy
        f5 = mu * (u_y + v_x) - sig_xy
        pde_score = torch.square(f0) + torch.square(f1) + torch.square(f2) + torch.square(f3) + torch.square(f4) + torch.square(f5)
        #sort dastaset base on pde_score in asscending way
        dataset = torch.concat([x_,pde_score], axis=1)
        last_column = dataset[:, -1]
        sorted_indices = torch.argsort(last_column)
        split_indices = torch.split(sorted_indices, torch.unique(last_column, sorted=False, return_counts=True)[1].tolist())

        sorted_data = torch.cat([torch.index_select(dataset, 0, indices) for indices in split_indices])
        total_points = int(len(sorted_data) - percent*len(sorted_data))
        chosen_points = sorted_data[total_points:].clone()
        return chosen_points[:,:2]


    def pde_loss_conduction(self,x_):
        x = x_.clone()
        x.requires_grad = True
        T = self.predict_conduction(x)
        T_out = grad(T.sum(), x,create_graph=True)[0]
        T_x = T_out[:, 0:1]
        T_y = T_out[:, 1:2]
        T_x_out = grad(T_x.sum(), x, create_graph=True)[0]
        T_xx = T_x_out[:, 0:1]
        del T_x_out
        T_y_out = grad(T_y.sum(), x, create_graph=True)[0]
        T_yy = T_y_out[:, 1:2]
        del T_y_out
        f0 = T_xx + T_yy
        mse_f0 = torch.mean(torch.square(f0))
        mse_pde = mse_f0 
        return mse_pde
    
    def point_selection_conduction(self,x_,percent=0.1):
        x = x_.clone()
        x.requires_grad = True
        T = self.predict_conduction(x)
        T_out = grad(T.sum(), x,create_graph=True)[0]
        T_x = T_out[:, 0:1]
        T_y = T_out[:, 1:2]
        T_x_out = grad(T_x.sum(), x, create_graph=True)[0]
        T_xx = T_x_out[:, 0:1]
        del T_x_out
        T_y_out = grad(T_y.sum(), x, create_graph=True)[0]
        T_yy = T_y_out[:, 1:2]
        del T_y_out
        f0 = T_xx + T_yy 
        pde_score = torch.square(f0) 
        #sort dastaset base on pde_score in asscending way
        dataset = torch.concat([x_,pde_score], axis=1)
        last_column = dataset[:, -1]
        sorted_indices = torch.argsort(last_column)
        split_indices = torch.split(sorted_indices, torch.unique(last_column, sorted=False, return_counts=True)[1].tolist())
        sorted_data = torch.cat([torch.index_select(dataset, 0, indices) for indices in split_indices])
        total_points = int(len(sorted_data) - percent*len(sorted_data))
        chosen_points = sorted_data[total_points:].clone()
        return chosen_points[:,:2]
    
    def pde_loss_burger(self,x1,viscosity=0.01):
        x = x1.clone()
        x.requires_grad = True
        u = self.predict_burger(x)
        u_out = grad(u.sum(), x,create_graph=True)[0]
        u_x = u_out[:,0:1]
        u_t = u_out[:,1:2]
        u_x_out = grad(u_x.sum(), x, create_graph=True)[0]
        u_xx = u_x_out[:,0:1]
        del u_x_out
        f0 = u_t + u*u_x - viscosity*u_xx
        mse_f0 = torch.mean(torch.square(f0))
        return mse_f0
    def point_selection_burger(self,x_,viscosity=0.01,percent=0.1):
        x = x_.clone()
        x.requires_grad = True
        u = self.predict_burger(x)
        u_out = grad(u.sum(), x,create_graph=True)[0]
        u_x = u_out[:,0:1]
        u_t = u_out[:,1:2]
        u_x_out = grad(u_x.sum(), x, create_graph=True)[0]
        u_xx = u_x_out[:,0:1]
        del u_x_out
        f0 = u_t + u*u_x - viscosity*u_xx
        pde_score = torch.square(f0) 
        dataset = torch.concat([x_,pde_score], axis=1)
        last_column = dataset[:, -1]
        sorted_indices = torch.argsort(last_column)
        split_indices = torch.split(sorted_indices, torch.unique(last_column, sorted=False, return_counts=True)[1].tolist())
        sorted_data = torch.cat([torch.index_select(dataset, 0, indices) for indices in split_indices])
        total_points = int(len(sorted_data) - percent*len(sorted_data))
        chosen_points = sorted_data[total_points:].clone()
        return chosen_points[:,:2]
        
    def bc_loss_navier(self, x_bd, value_bnd):
        u, v = self.predict_navier(x_bd)[0:2]
        mse_bc = torch.mean(torch.square(u - value_bnd[:, 0:1])) + torch.mean(torch.square(v - value_bnd[:, 1:2]))
        return mse_bc

    def outlet_loss_navier(self, x):
        p,_ = self.predict_navier(x)[2:4]
        mse_outlet = torch.mean(torch.square(p))
        return mse_outlet

    def bc_loss_prediction(self, x_bd, value_bnd):
        T = self.predict_burger(x_bd)
        mse_bc = torch.mean(torch.square(T - value_bnd)) 
        return mse_bc 



    def adam_optimizer(self,EPOCHS,xy_col,xy_bd,uv_bd,threshold = 1e-20,landa=1):
        '''
        Training on the Primary Loss Function for the first benchmark introduced in the paper 
        '''
        losse_bc= []
        losses_pde = []
        if self.PDE == 'Navier_Stokes':
            for epoch in tqdm(range(EPOCHS)):
                self.optim.zero_grad()
                mse_bc = self.bc_loss_navier(xy_bd,uv_bd)
                mse_pde = self.pde_loss_navier(xy_col)
                loss = mse_bc + mse_pde 
                loss.backward()
                losse_bc.append(mse_bc.detach().cpu().item())
                losses_pde.append(mse_pde.detach().cpu().item())
                self.optim.step()
                if (epoch+1) % 1000 == 0:
                    print('Epoch: {}, Loss: {}, bc: {}, PDE: {}'.format(epoch, loss.item(),mse_bc.item(),mse_pde.item()))
        elif self.PDE == 'Heat_Conduction':
            for epoch in tqdm(range(EPOCHS)):
                self.optim.zero_grad()
                mse_bc = self.bc_loss_prediction(xy_bd,uv_bd)
                mse_pde = self.pde_loss_conduction(xy_col)
                loss = landa*mse_bc + mse_pde 
                loss.backward()
                losse_bc.append(mse_bc.detach().cpu().item())
                losses_pde.append(mse_pde.detach().cpu().item())
                self.optim.step()
                if (epoch+1) % 1000 == 0:
                    print('Epoch: {}, Loss: {}, bc: {}, PDE: {}'.format(epoch, loss.item(),mse_bc.item(),mse_pde.item()))
                if loss.detach().cpu().item()  <= threshold:
                    return losse_bc,losses_pde,self.pinn        
        elif self.PDE == 'Burger':
            best_loss = 1000000
            best_epoch = 0
            for epoch in tqdm(range(EPOCHS)):
                self.optim.zero_grad()
                mse_bc = self.bc_loss_prediction(xy_bd,uv_bd)
                mse_pde = self.pde_loss_burger(xy_col)
                loss = landa*mse_bc + mse_pde 
                loss.backward()
                losse_bc.append(mse_bc.detach().cpu().item())
                losses_pde.append(mse_pde.detach().cpu().item())
                self.optim.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch                    
                if (epoch+1) % 1000 == 0:
                    print('Epoch: {}, Loss: {}, bc: {}, PDE: {}'.format(epoch, loss.item(),mse_bc.item(),mse_pde.item()))
                if loss.detach().cpu().item()  <= threshold:
                    return losse_bc,losses_pde,self.pinn
                if loss.detach().cpu().item()  >= 1000:
                    print('Model Diverged')
                    return losse_bc,losses_pde,self.pinn

                if epoch - best_epoch > 10000:
                    print('No improvement in 10000 epochs, stopping')
                    return losse_bc,losses_pde,self.pinn

        else:
            raise ValueError("Unsupported PDE type") 

        return losse_bc,losses_pde,self.pinn
        
    def lbfgs_optimizer(self,Epochs,xy_col,xy_bd,uv_bd,outlet_xy=None,num_iterations=1,landa=1,threshold = 0.0001):
        '''
        Second training loop
        If you use proposed method in the article you can keep landa at 1 o.w. tune it 
        '''
        total_loss = []

        optimizer = torch.optim.LBFGS(self.pinn.parameters(),lr=self.lr, max_iter=num_iterations)
        if self.PDE == 'Navier_Stokes':
            def closure():
                optimizer.zero_grad()
                mse_bc = self.bc_loss_navier(xy_bd,uv_bd) + self.outlet_loss_navier(outlet_xy)
                mse_pde = self.pde_loss_navier(xy_col) 
                loss = mse_pde + landa*mse_bc
                loss.backward()
                return loss

            for i in tqdm(range(Epochs)):
                loss = optimizer.step(closure)
                total_loss.append(loss.detach().cpu().item())
                if loss.detach().cpu().item()  <= threshold:
                    return total_loss,self.pinn
                if i % 100 == 0:
                    print('Epoch: {}, Loss: {}'.format(i, loss.item()))
        
        elif self.PDE == 'Heat_Conduction':
            def closure():
                optimizer.zero_grad()
                mse_bc = self.bc_loss_prediction(xy_bd,uv_bd)
                mse_pde = self.pde_loss_conduction(xy_col) 
                loss = mse_pde + landa*mse_bc
                loss.backward()
                return loss
            for i in tqdm(range(Epochs)):
                loss = optimizer.step(closure)
                total_loss.append(loss.detach().cpu().item())
                if loss.detach().cpu().item()  <= threshold:
                    return total_loss,self.pinn
                if i % 100 == 0:
                    print('Epoch: {}, Loss: {}'.format(i, loss.item()))
        elif self.PDE == 'Burger':
            def closure():
                optimizer.zero_grad()
                mse_bc = self.bc_loss_prediction(xy_bd,uv_bd) 
                mse_pde = self.pde_loss_burger(xy_col) 
                loss = mse_pde + landa*mse_bc
                loss.backward()
                return loss

            for i in tqdm(range(Epochs)):
                loss = optimizer.step(closure)
                total_loss.append(loss.detach().cpu().item())
                if loss.detach().cpu().item()  <= threshold:
                    return total_loss,self.pinn
                if i % 100 == 0:
                    print('Epoch: {}, Loss: {}'.format(i, loss.item()))        
        else:
            raise ValueError("Unsupported PDE type")

        return total_loss,self.pinn

    def lbfgs_optimizer_inverse(self,Epochs,xy_col,xy_inverse,uv_inverse,xy_bd,uv_bd,outlet_xy,num_iterations=1,landa=1):
        total_loss = []
        optimizer = torch.optim.LBFGS(self.pinn.parameters(),lr=self.lr, max_iter=num_iterations)
        best_loss = 1000000
        best_epoch = 0
        def closure():
            optimizer.zero_grad()
            mse_bc = self.bc_loss_navier(xy_bd,uv_bd) + self.outlet_loss_navier(outlet_xy)
            loss_inverse = self.bc_loss_navier(xy_inverse,uv_inverse)
            mse_pde = self.pde_loss_navier(xy_col)
            loss = + landa*mse_bc + loss_inverse + mse_pde 
            loss.backward()
            return loss

        for i in tqdm(range(Epochs)):
            loss = optimizer.step(closure)
            total_loss.append(loss.detach().cpu().item())
            if loss.detach().cpu().item()  <= 1e-4:
                return total_loss,self.pinn
            if i % 100 == 0:
                print('Epoch: {}, Loss: {}'.format(i, loss.item()))
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = i  
            if i - best_epoch > 10000:
                print('No improvement in 10000 epochs, stopping')
                return total_loss,self.pinn
            if loss.detach().cpu().item()  >= 1000:
                print('Model Diverged')
                return total_loss,self.pinn
        return total_loss,self.pinn

    def compelete_train_inversed(self,EPOCHS,xy_col, x_inverse,value_inverse, x_bd, value_bnd):
        'first phase'

        train_loss = []
        epochs = EPOCHS
        for epoch in tqdm(range(epochs)):
            self.optim.zero_grad()
            mse_bc = self.bc_loss_navier(x_bd, value_bnd)
            mse_pde = self.pde_loss_navier(xy_col) 
            mse_inverse_ = self.bc_loss_navier(x_inverse,value_inverse)
            loss = mse_pde+ mse_bc +  mse_inverse_
            loss.backward()
            train_loss.append(loss.detach().cpu().item())
            self.optim.step()
            if (epoch+1) % 1000 == 0:
                print('Epoch: {}, Loss: {}, inverse:{},mse:{}'.format(epoch, loss.item(),mse_inverse_.item(),mse_bc.item()))

        return train_loss,self.pinn