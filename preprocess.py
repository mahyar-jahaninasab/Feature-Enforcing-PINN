from model import * 
from training_manager import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class batch_maker:
    def __init__(self,cylinder=[0,0,0.1],domain_ub=[1,1],domain_lb=[-1,-1],A=1,Centered=True):
        super().__init__()
        '''
            A: total amount of domain expansion around the cylinder
            A=1 corrsponds to a square that exactly matches the cylinder
        '''
        self.centered = Centered
        self.x_cyc,self.y_cyc,self.r_cyc= cylinder[0],cylinder[1],cylinder[2]
        self.upper_bound_cyl = np.array([self.x_cyc + self.r_cyc, self.y_cyc +  self.r_cyc])
        self.lower_bound_cyl = np.array([self.x_cyc - self.r_cyc, self.y_cyc -  self.r_cyc])
        self.sampler = qmc.LatinHypercube(d=2, strength=2 ,optimization="random-cd")
        self.y_ub,self.y_lb = domain_ub[1],domain_lb[1]
        self.x_ub,self.x_lb = domain_ub[0],domain_lb[0]
        self.A = A
        assert A >= 1
        assert self.x_cyc >= 0
        assert self.y_cyc >= 0
        assert self.r_cyc > 0
        assert 2*self.r_cyc < self.y_ub - self.y_lb
        #check these methods Sobol', Halton,
    def cylinder2d(self,N_cyc):
        '''
            Defining a 2D cylinder.
            x_c & y_c shows the center of the cylinder
            r: repreasents circle radius
            N is the total number of boundary points on cylinder
            The default value for x_c,y_c,r = [0,0,0.1]
        '''
        assert N_cyc >= 1
        teta = 2 * np.random.random(size= N_cyc) - 1
        #To make sure these values  always exist and not repeated
        mask = (teta != 0) & (teta != 0.5) & (teta != 1) & (teta != 1.5)
        teta = teta[mask]
        teta = np.append(teta,[0,0.5,1,1.5])
        obstacle_x = (self.r_cyc * np.cos(teta*np.pi)+ self.x_cyc).reshape(-1, 1)
        obstacle_y = (self.r_cyc * np.sin(teta*np.pi) + self.y_cyc).reshape(-1, 1)
        obstacle_xy = np.concatenate([obstacle_x, obstacle_y], axis=1)
        return obstacle_xy

    def around_obstacle(self,N_a):
        '''
            Defines new points around cylinder
            N_a: total points around cyrcle
        '''
        around_obstacle_data = self.sampler.random(N_a)
        around_obstacle_data = qmc.scale(around_obstacle_data, self.A*self.lower_bound_cyl, self.A*self.upper_bound_cyl)
        mask_out_cyl = np.sqrt((around_obstacle_data[:, 0] - self.x_cyc) ** 2 +
                                   (around_obstacle_data[:, 1] - self.y_cyc) ** 2)
        around_obstacle_data = around_obstacle_data[mask_out_cyl > self.r_cyc].reshape(-1, 2)

        return around_obstacle_data

    def design_batches(self,number_of_batches, N_b,N,shared_points=False,N_shared_b=0,N_shared =0,N_square =0,S=0.99):

        '''
            Splitting dataset into designed batches
            number_of_batches refers to total number of bathces
            N_b refers to total number of points in each batch
            S is a hyperparameter that excludes boundary points
        '''
        y_ub_domain = S*self.y_ub - self.A*self.upper_bound_cyl[1]
        y_lb_domain = abs(S*self.y_lb - self.A*self.lower_bound_cyl[1])
        x_ub_domain = S*self.x_ub - self.A*self.upper_bound_cyl[0]
        x_lb_domain = abs(S*self.x_lb - self.A*self.lower_bound_cyl[0])

        splitted_y_ub = y_ub_domain/number_of_batches
        splitted_y_lb = y_lb_domain/number_of_batches
        splitted_x_ub = x_ub_domain/number_of_batches
        splitted_x_lb = x_lb_domain/number_of_batches

        dataset = []
        if shared_points:
          shared_data_over_all_batches = []
        if self.centered :
            #doesnt work if lower domain is positive
            right_u_bound = np.array([S*self.x_ub , S*self.y_ub - splitted_y_ub])
            right_l_bound = np.array([S*self.x_ub - splitted_x_ub, S*self.y_lb  + splitted_y_lb])
            left_l_bound = np.array([S*self.x_lb , S*self.y_lb + splitted_y_lb])
            left_u_bound = np.array([S*self.x_lb + splitted_x_lb, S*self.y_ub  - splitted_y_ub])
            up_u_bound = np.array([S*self.x_ub , S*self.y_ub])
            up_l_bound = np.array([S*self.x_lb , S*self.y_ub  - splitted_y_ub])
            down_u_bound = np.array([S*self.x_ub , S*self.y_lb + splitted_y_lb])
            down_l_bound = np.array([S*self.x_lb, S*self.y_lb ])


            for i in range(number_of_batches):
                data = self.sampler.random(N_b)
                data_ = self.sampler.random(N)
                data_r = qmc.scale(data_, right_l_bound, right_u_bound)
                data_l = qmc.scale(data_, left_l_bound, left_u_bound)
                data_u = qmc.scale(data, up_l_bound, up_u_bound)
                data_d = qmc.scale(data, down_l_bound, down_u_bound)
                new_data = np.concatenate([data_r, data_l,data_u,data_d], axis=0)
                dataset.append(new_data)

                if shared_points:
                  data = self.sampler.random(N_shared_b)
                  data_ = self.sampler.random(N_shared)
                  data_r = qmc.scale(data_, right_l_bound, right_u_bound)
                  data_l = qmc.scale(data_, left_l_bound, left_u_bound)
                  data_u = qmc.scale(data, up_l_bound, up_u_bound)
                  data_d = qmc.scale(data, down_l_bound, down_u_bound)
                  new_data = np.concatenate([data_r, data_l,data_u,data_d], axis=0)
                  shared_data_over_all_batches.append(new_data)

                right_u_bound[0] -= splitted_x_ub
                right_u_bound[1] -= splitted_y_ub
                right_l_bound[0] -= splitted_x_ub
                right_l_bound[1] += splitted_y_lb
                left_u_bound[0] += splitted_x_lb
                left_u_bound[1] -= splitted_y_ub
                left_l_bound[0] += splitted_x_lb
                left_l_bound[1] += splitted_y_lb
                up_u_bound[0] -= splitted_x_ub
                up_u_bound[1] -= splitted_y_ub
                up_l_bound[0] += splitted_x_lb
                up_l_bound[1] -= splitted_y_ub
                down_u_bound[0] -= splitted_x_ub
                down_u_bound[1] += splitted_y_lb
                down_l_bound[0] += splitted_x_lb
                down_l_bound[1] += splitted_y_lb
        else:
            right_u_bound = np.array([S*self.x_ub , S*self.y_ub])
            right_l_bound = np.array([S*self.x_ub - splitted_x_ub, S*self.y_lb])
            left_u_bound = np.array([S*self.x_lb + splitted_x_lb, S*self.y_ub])
            left_l_bound = np.array([S*self.x_lb , S*self.y_lb ])
            up_u_bound = np.array([self.A*self.upper_bound_cyl[0], S*self.y_ub])
            up_l_bound = np.array([self.A*self.lower_bound_cyl[0], self.A*self.upper_bound_cyl[1]])
            down_u_bound = np.array([self.A*self.upper_bound_cyl[0], self.A*self.lower_bound_cyl[1]])
            down_l_bound = np.array([self.A*self.lower_bound_cyl[0], S*self.y_lb ])
            data = self.sampler.random(N_b)
            data_up = qmc.scale(data, up_l_bound, up_u_bound)
            dataset.append(data_up)
            data_down = qmc.scale(data, down_l_bound, down_u_bound)
            dataset.append(data_down)
            for i in range(number_of_batches):
                data = self.sampler.random(N_b)
                data_right = qmc.scale(data, right_l_bound, right_u_bound)
                data_left = qmc.scale(data, left_l_bound, left_u_bound)
                dataset.append(data_right)
                dataset.append(data_left)
                right_l_bound[0] -= splitted_x_ub
                right_u_bound[0] -= splitted_x_ub
                left_u_bound[0]  += splitted_x_lb
                left_l_bound[0]  += splitted_x_lb

        if shared_points:
          square = self.around_obstacle(N_square)
          data__ = []
          temp_data_ = np.concatenate(shared_data_over_all_batches[:], axis=0).reshape(-1,2)
          for i in range(number_of_batches):
            temp_array = np.concatenate([dataset[i], temp_data_,square],axis=0)
            temp_array = np.unique(temp_array, axis=0)
            data__.append(temp_array)
          return data__
        else:
          return dataset

    def boundary(self,N_boundary,N_wall,value_u=1,value_v=0,value_outlet=0):
        inlet_x = np.zeros((N_boundary, 1))
        inlet_y = np.random.uniform(self.y_lb, self.y_ub, (N_boundary, 1))
        #change ThIS PART LATER
        #inlet_u = value_u*np.ones((N_boundary, 1))
        inlet_u = 4 * inlet_y * (0.4 - inlet_y) / (0.4 ** 2)
        inlet_v = value_v*np.ones((N_boundary, 1))
        inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
        inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)
        #prssure=0 in outlet
        outlet_xy = np.random.uniform([self.x_ub, self.y_lb], [self.x_ub, self.y_ub], (N_boundary, 2))
        outlet_value = value_outlet*np.ones((N_boundary, 1))
        #walls with no slip conditions
        upwall_xy = np.random.uniform([self.x_lb, self.y_ub], [self.x_ub, self.y_ub], (N_wall, 2))
        dnwall_xy = np.random.uniform([self.x_lb,  self.y_lb], [self.x_ub, self.y_lb], (N_wall, 2))
        upwall_uv = np.zeros((N_wall, 2))
        dnwall_uv = np.zeros((N_wall, 2))
        wall_xy = np.concatenate([upwall_xy, dnwall_xy], axis=0)
        wall_uv = np.concatenate([upwall_uv, dnwall_uv], axis=0)
        return wall_xy,wall_uv,outlet_xy,outlet_value,inlet_xy,inlet_uv
    def dataset(self,*args):
        number_of_batches,N_b,S = args[0],args[1],args[2]
        number_of_batches,N_b,S = args[0],args[1],args[2]
        data_domain = self.design_batches(number_of_batches, N_b,S)
        boundary = []
        '''for i in range number_of_batches:
            boundary(self,N_boundary,N_wall,value_u=1,value_v=0,value_outlet=0):'''
    def plot(self,data,title='title'):
        plt.title(title)
        plt.scatter(data[:, 0], data[:, 1], s=.2, marker=".", c="r", label="CP")
        plt.show()

    def plot1(self, data, title='Title', xlabel='X-axis label', ylabel='Y-axis label'):
      fig, ax = plt.subplots(figsize=(6, 6))
      ax.scatter(data[:, 0], data[:, 1], s=2, marker="o", c="red", label="CP")
      ax.set_xlabel(xlabel, fontsize=12)
      ax.set_ylabel(ylabel, fontsize=12)
      ax.set_title(title, fontsize=14)
      ax.tick_params(axis='both', which='major', labelsize=10)
      ax.grid(True)
      plt.tight_layout()
      plt.show()

def pipe_line(neurons_in_layaers,tensor_total_domain,ub,lb,n=10,PDE='Navier_Stokes'): 
    '''
    Reducing Variance and Preparing Inputs for Training on the Primary Loss Function using the intial state of different neural networks
    n is a variance reduction factor. For instance, n = 10 means the variance will be reduced by the square root of ten.
    We reduced the variance by averaging out the initial state distribution of n neural networks.
    '''
    state_model = []
    total = []
    new_model = {}
    LR=None
    for _ in range(n):
        chosen_points_ = []
        model = PINN_Net(PDE,neurons_in_layaers,ub,lb,nn.Tanh()).to(DEVICE)
        state_model.append(model.state_dict())
        trainer = PINNManager(model,LR,PDE)
        for j in range(len(tensor_total_domain)):
            if PDE == 'Navier_Stokes':
                chosen_points_.append(trainer.point_selection_navier(tensor_total_domain[j].to(DEVICE),percent=0.1))
            elif PDE == 'Heat_Conduction':
                chosen_points_.append(trainer.point_selection_conduction(tensor_total_domain[j].to(DEVICE),percent=0.1))
            elif PDE == 'Burger':
                chosen_points_.append(trainer.point_selection_burger(tensor_total_domain[j].to(DEVICE),percent=0.1))
            else:
                raise ValueError("Unsupported PDE type")  
        total.append(chosen_points_)
        del model
        del  chosen_points_
    for key in state_model[0].keys():
        new_model[key] = 0

    for i in range(len(state_model)):
        for key in state_model[0].keys():
            new_model[key]+= state_model[i][key]

    for key in state_model[0].keys():
        new_model[key] /=  len(state_model)
    
    temp = [torch.from_numpy(np.concatenate([total[i][j].detach().cpu().numpy() for i in range(n)], axis=0)) for j in range(len(tensor_total_domain))]
    result = torch.cat(temp, dim=0)
    result = torch.unique(result, dim=0).to(DEVICE)

    return new_model,result

def load_smart_weights(complete_model,smart_weight):
    '''
    Please note that `copy.deepcopy` is not used, so the previous state will also be modified after the final training.

    '''
    train_key = list(complete_model.state_dict())
    sub_train_key = list(smart_weight.state_dict())
    state_warmed = []
    state_last =   []
    state_warmed.append(smart_weight.state_dict())
    state_last.append(complete_model.state_dict())
    last_state = {}
    for key in train_key:
        last_state[key] = 0
    for key in train_key:
        last_state[key] = state_last[0][key]
    for key in sub_train_key:
        last_state[key] = state_warmed[0][key]
    return last_state



def prepare_data(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.4, r=0.05, xc=0.2, yc=0.2, N_b=200, N_w=400, N_s=200, N_c=40000, N_r=10000):
    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])

    def getData():
        # inlet, v=0 & inlet velocity
        inlet_x = np.zeros((N_b, 1))
        inlet_y = np.random.uniform(y_min, y_max, (N_b, 1))
        inlet_u = 4 * inlet_y * (0.4 - inlet_y) / (0.4 ** 2)
        inlet_v = np.zeros((N_b, 1))
        inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
        inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

        # outlet, p=0
        xy_outlet = np.random.uniform([x_max, y_min], [x_max, y_max], (N_b, 2))

        # wall, u=v=0
        upwall_xy = np.random.uniform([x_min, y_max], [x_max, y_max], (N_w, 2))
        dnwall_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
        upwall_uv = np.zeros((N_w, 2))
        dnwall_uv = np.zeros((N_w, 2))

        # cylinder surface, u=v=0
        theta = np.linspace(0.0, 2 * np.pi, N_s)
        cyl_x = (r * np.cos(theta) + xc).reshape(-1, 1)
        cyl_y = (r * np.sin(theta) + yc).reshape(-1, 1)
        cyl_xy = np.concatenate([cyl_x, cyl_y], axis=1)
        cyl_uv = np.zeros((N_s, 2))

        # all boundary except outlet
        xy_bnd = np.concatenate([inlet_xy, upwall_xy, dnwall_xy, cyl_xy], axis=0)
        uv_bnd = np.concatenate([inlet_uv, upwall_uv, dnwall_uv, cyl_uv], axis=0)

        # Collocation
        xy_col = lb + (ub - lb) * lhs(2, N_c)

        # refine points around cylider
        refine_ub = np.array([xc + 2 * r, yc + 2 * r])
        refine_lb = np.array([xc - 2 * r, yc - 2 * r])

        xy_col_refine = refine_lb + (refine_ub - refine_lb) * lhs(2, N_r)
        xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

        # remove collocation points inside the cylinder
        dst_from_cyl = np.sqrt((xy_col[:, 0] - xc) ** 2 + (xy_col[:, 1] - yc) ** 2)
        xy_col = xy_col[dst_from_cyl > r].reshape(-1, 2)

        # concatenate all xy for collocation
        xy_col = np.concatenate((xy_col, xy_bnd, xy_outlet), axis=0)

        # convert to tensor
        xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(DEVICE)
        uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(DEVICE)
        xy_outlet = torch.tensor(xy_outlet, dtype=torch.float32).to(DEVICE)
        xy_col = torch.tensor(xy_col, dtype=torch.float32).to(DEVICE)
        return xy_col.to(DEVICE), xy_bnd.to(DEVICE), uv_bnd.to(DEVICE), xy_outlet.to(DEVICE)

    return getData()



class InputGenerator_2D_heatconduction:
    def __init__(self, num_boundary_conditions=4, num_data_per_condition=100, boundary_values=[265., 300., 400., 273.]):
        max_temp = max(boundary_values)
        boundary_values = [i/max_temp for i in boundary_values]
        self.num_boundary_conditions = num_boundary_conditions
        self.num_data_per_condition = num_data_per_condition
        self.boundary_values = boundary_values

    def generate_data(self):
        engine = qmc.LatinHypercube(d=1)
        data = np.zeros([self.num_boundary_conditions, self.num_data_per_condition, 3])

        for i, j in zip(range(self.num_boundary_conditions), [-1, +1, -1, +1]):
            points = (engine.random(n=self.num_data_per_condition)[:, 0] - 0.5) * 2
            if i < 2:
                data[i, :, 0] = j
                data[i, :, 1] = points
            else:
                data[i, :, 0] = points
                data[i, :, 1] = j

        for i in range(self.num_boundary_conditions):
            data[i, :, 2] = self.boundary_values[i]

        data = data.reshape(self.num_data_per_condition * self.num_boundary_conditions, 3)
        return data

    def generate_collocation_points(self, num_collocation_points=40000):
        engine = qmc.LatinHypercube(d=2)
        collocation_points = engine.random(n=num_collocation_points)
        collocation_points = 1.999 * (collocation_points -0.5) 
        return collocation_points


def get_burger_boundaries(bonds=[0,4,0,5],num_samples = 30,time_grid=0.01,boundary_grid=0.0013334):
    x_min = bonds[0]
    x_max = bonds[1]
    T_min = bonds[2]
    T_max = bonds[3]

    engine = qmc.LatinHypercube(d=2)
    samples = engine.random(n=num_samples)
    x_samples = samples[:, 0] * (x_max - x_min) + x_min  # x_min = 0, x_max = 4

    T_samples =  np.arange(T_min,T_max, time_grid)
    X, T = np.meshgrid(x_samples, T_samples)
    x = X.reshape(-1, 1)
    t = T.reshape(-1, 1)
    xt = np.concatenate([x, t], axis=1)
    xt_domain = torch.tensor(xt, dtype=torch.float32).to(DEVICE)
    x_ = np.arange(x_min,x_max, boundary_grid)
    X_initial, T_initial = np.meshgrid(x_, T_min)
    x_initial = X_initial.reshape(-1, 1)
    t_initial = T_initial.reshape(-1, 1)
    xt_initial = np.concatenate([x_initial, t_initial], axis=1)
    xt_intial = torch.tensor(xt_initial, dtype=torch.float32).to(DEVICE)
    #boundary for different netwroks
    # b.c_left
    t_bcl = np.arange(T_min,T_max, time_grid)
    X_bcl, T_bcl = np.meshgrid(x_min, t_bcl)
    x_bcl = X_bcl.reshape(-1, 1)
    t_bcl = T_bcl.reshape(-1, 1)
    xt_bcl = np.concatenate([x_bcl, t_bcl], axis=1)
    xt_bcl = torch.tensor(xt_bcl, dtype=torch.float32).to(DEVICE)
    #bc-right
    t_bcr = np.arange(T_min,T_max, time_grid)
    X_bcr, T_bcr = np.meshgrid(x_max, t_bcr)
    x_bcr = X_bcr.reshape(-1, 1)
    t_bcr = T_bcr.reshape(-1, 1)
    xt_bcr = np.concatenate([x_bcr, t_bcr], axis=1)
    xt_bcr = torch.tensor(xt_bcr, dtype=torch.float32).to(DEVICE)

    return xt_domain,xt_intial,xt_bcr,xt_bcl

def burger_ans(x,t):
    nominator = 2*0.01*torch.pi*torch.sin(torch.pi*x)*torch.exp((-torch.pi**2)*(t-5)*0.01)
    denominator = 2 + torch.cos(torch.pi*x)*torch.exp((-torch.pi**2)*(t-5)*0.01)
    return nominator/denominator