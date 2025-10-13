
#--CLASS FOR FINDING VALUES FOR ONE LOOP WITH VARIOUS PARAMETERS--
class parameterGrid:
    def __init__(self, n_r, n_B, n_k, n_mu, r_min=R_min/R_max, r_max=1, b_min=B_min/B_max, b_max=1, dk_min=0.1, dk_max=2, mu_min=MU_min, mu_max=MU_max, k = True):
        self.nR = n_r
        self.nB = n_B
        self.nK = n_k
        self.nM = n_mu
        
        self.Rmin = r_min
        self.Rmax = r_max
        self.Bmin = b_min
        self.Bmax = b_max
        self.kmin = dk_min
        self.kmax = dk_max
        self.Mmin = mu_min
        self.Mmax = mu_max
        
        if k:
            self.fermi = kFAu
        else:
            self.fermi = 0.0
        
        #step size for grids???
        #print(self.fermi)
        self.buildGrid()
        
    def buildGrid(self):
        self.rlist = []
        self.blist = []
        self.klist = []
        self.mlist = []
        self.lgrid = [] #build as grid or list??? saved as list, so probably not needed to be a grid
        Rint = (self.Rmax-self.Rmin) / (self.nR -1) #start linear
        Bint = (self.Bmax-self.Bmin) / (self.nB -1) #start linear
        kint = (self.kmax-self.kmin) / (self.nK -1) #probably should be evenly spaced over 1 period to start
        mint = (np.log(self.Mmax)-np.log(self.Mmin)) / (self.nM -1) #log spacing?
        for Ri in range(self.nR):
            R  = self.Rmin + Rint * Ri
            for Bi in range(self.nB):
                B  = self.Bmin + Bint * Bi
                for ki in range(self.nK):
                    dk = self.kmin + kint * ki
                    for mi in range(self.nM):
                        mu = np.exp(np.log(self.Mmin) + mint * mi) #log spacing?
                        f  = self.fermi
                        self.rlist.append(R*R_max)
                        self.blist.append(B*B_max)
                        self.klist.append(dk)
                        self.mlist.append(mu)
                        #print(R, B, dk, mu)
                        self.lgrid.append(loop(R,B,dk,mu,f))
                        #print(self.lgrid[0].B)
                                                
    def ivp_solve(self, per_range = 1.0, method = 'RK45'):
        for i in range(len(self.lgrid)):
            #print(self.lgrid[i].R)
            self.lgrid[i].solve_ivp(ee_int = True, percent_range=per_range, method = method)
    
    def bvp_solve(self, gs= 10, per_range = 1.0, method = 'RK45', mamp = 1000.):
        self.gs = gs
        for i in range(len(self.lgrid)):
            print(i)
            self.lgrid[i].derivGrid(self, grid_size = gs,  pr=per_range, method = method, max_amp = mamp)

        #for i in range(len(self.lgrid)):
        #self.lgrid[i].findDeriv(self, pr = per_range, method = method, max_amp = mamp)

        
    def save_data_bvp(self, n0 = 20):
        #get the rest of the data 
        #for the original solution
        self.pf_list = []
        self.x0_list = []
        self.pdf_list = []
        self.pd0_list = []
        #self.ftlist = []
        #self.fylist = []
        #self.fdlist = []
        for i in range(n0):
            self.x0_list.append([])
        for i in range(len(self.lgrid)):
            self.pdf_list.append(self.lgrid[i].sol["y"][1,-1])
            self.pf_list.append(self.lgrid[i].sol["y"][0, -1])
            self.pd0_list.append(self.lgrid[i].sol["y"][1,0])
            #when real(psi) = 0
            for j in range(n0):
                if len(self.lgrid[i].sol["t_events"]) < j:
                    self.x0_list[j].append(self.lgrid[i].sol["t_events"][j])
                else:
                    self.x0_list[j].append(-1.0)
            #full solution
                  
        #get the data for the grid
        for i in range(len(self.lgrid)):
            for j in range(self.gs):
                for k in range(self.gs):
                    self.rlist.append(self.rlist[i])
                    self.blist.append(self.blist[i])
                    self.klist.append(self.klist[i])
                    self.mlist.append(self.mlist[i])
                    self.pf_list.append(self.lgrid[i].y_grid[j,k])
                    self.pdf_list.append(self.lgrid[i].d_grid[j,k])
                    self.pd0_list.append(self.lgrid[i].d0_grid[j,k])
                    #when real(psi) = 0
                    for ii in range(n0):
                        if(len(self.lgrid[i].s_grid[j][k]['t_events'])<ii):
                            self.x0_list[ii].append(self.lgrid[i].s_grid[j][k]['t_events'][ii])
                        else:
                            self.x0_list[ii].append(-1.0)
                    #full solution
                    #self.ftlist.append(self.lgrid[i].s_grid[j][k]['t'])#[ii]
                    #self.lgrid[i].s_grid[j][k]['y'][0,ii]
                    #self.lgrid[i].s_grid[j][k]['y'][1,ii]
                    
        
        #create a dataframe to save
        d = {'R': self.rlist, 'B': self.blist, 'dk': self.klist, 'mu': self.mlist, 'psi_d_0': self.pd0_list, 'psi_f': self.pf_list,'psi_d_f': self.pdf_list}
        for i in range(n0):
            d[f"x{i+1}"] = self.x0_list[i]
        #and add the full solutions
        self.df = pd.DataFrame(data=d)
        
        #save as csv
        self.df.to_csv('grid_0001.csv') #change file name!!!
        
    def save_data(self, n0 = 20):
        #get the rest of the data
        self.pf_list = []
        self.x0_list = []
        self.pdf_list = []
        self.pd0_list = []
        for i in range(n0):
            self.x0_list.append([])
        for i in range(len(self.lgrid)):
            self.pdf_list.append(self.lgrid[i].sol["y"][1,-1])
            self.pf_list.append(self.lgrid[i].sol["y"][0, -1])
            self.pd0_list.append(self.lgrid[i].sol["y"][1,0])
            #print(len(self.lgrid[i].sol["t_events"][0]))
            for j in range(n0):
                #print(i, " ", j, len(self.lgrid[i].sol["t_events"]))
                if len(self.lgrid[i].sol["t_events"][1]) > j:
                    self.x0_list[j].append(self.lgrid[i].sol["t_events"][1][j])
                else:
                    self.x0_list[j].append(-1.0)

        #create a dataframe to save
        d = {'R': self.rlist, 'B': self.blist, 'dk': self.klist, 'mu': self.mlist, 'psi_d_0': self.pd0_list, 'psi_f': self.pf_list,'psi_d_f': self.pdf_list}
        for i in range(n0):
            d[f"x{i+1}"] = self.x0_list[i]
        self.df = pd.DataFrame(data=d)
        
        #save as csv
        self.df.to_csv(f'grid_k_F_{k}.csv') #change file name!!!

    #Does non-linear effect shield drop???
