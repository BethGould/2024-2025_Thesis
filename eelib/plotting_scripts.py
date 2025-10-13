    
    
# plotting the grid
def plot_img(loopObj, i, j, k, R, B, mu):
        #position arrays
        tu = loopObj.s_grid_u[i][j]['t'][:]
        tl = loopObj.s_grid_l[i][j]['t'][:]
        #value arrays
        sui = np.imag(loopObj.s_grid_u[i][j]["y"][0, :])
        sli = np.imag(loopObj.s_grid_l[i][j]["y"][0, :])
        suo = np.imag(loopObj.s_grid_0u[i][j]["y"][0, :])
        slo = np.imag(loopObj.s_grid_0l[i][j]["y"][0, :])
        su2= np.imag(loopObj.psij(tu))
        sl2= np.imag(loopObj.psij(tl))


        #--PLOTTING--
        fig, ax = plt.subplots()
        ax.set_ylabel('Imaginary Part of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, dk={k}, R={R}, B={B}, \u03BC={mu}")
        line1, = ax.plot(tu, sui, color = 'red', label = 'with e-e interaction')
        line2, = ax.plot(tu, suo, color = 'green', label = 'without e-e interaction')
        line3, = ax.plot(tu, su2, color = 'blue', label = 'exact solution')
        line4, = ax.plot(tl, sli, color = 'red', label = 'with e-e interaction')
        line5, = ax.plot(tl, slo, color = 'green', label = 'without e-e interaction')
        line6, = ax.plot(tl, sl2, color = 'blue', label = 'exact solution')
        ax.legend(handles=[line1, line2, line3])
        ax.set_box_aspect(2.0/3.5)
        
        #plt.show()
        plt.savefig(f"plot_{i}_{j}_imag.pdf", format='pdf')
        plt.close()


def plot_real(loopObj, i, j, k, R, B, mu):
        #position arrays
        tu = loopObj.s_grid_u[i][j]['t'][:]
        tl = loopObj.s_grid_l[i][j]['t'][:]
        #value arrays
        sui = np.real(loopObj.s_grid_u[i][j]["y"][0, :])
        sli = np.real(loopObj.s_grid_l[i][j]["y"][0, :])
        suo = np.real(loopObj.s_grid_0u[i][j]["y"][0, :])
        slo = np.real(loopObj.s_grid_0l[i][j]["y"][0, :])
        su2= np.real(loopObj.psij(tu))
        sl2= np.real(loopObj.psij(tl))


        #--PLOTTING--
        fig, ax = plt.subplots()
        ax.set_ylabel('Real Part of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, dk={k}, R={R}, B={B}, \u03BC={mu}")
        line1, = ax.plot(tu, sui, color = 'red', label = 'with e-e interaction')
        line2, = ax.plot(tu, suo, color = 'green', label = 'without e-e interaction')
        line3, = ax.plot(tu, su2, color = 'blue', label = 'exact solution')
        line4, = ax.plot(tl, sli, color = 'red', label = 'with e-e interaction')
        line5, = ax.plot(tl, slo, color = 'green', label = 'without e-e interaction')
        line6, = ax.plot(tl, sl2, color = 'blue', label = 'exact solution')
        ax.legend(handles=[line1, line2, line3])
        ax.set_box_aspect(2.0/3.5)
        
        #plt.show()
        plt.savefig(f"plot_{i}_{j}_real.pdf", format='pdf')
        plt.close()

def plot_abs(loopObj, i, j, k, R, B, mu):
        #position arrays
        tu = loopObj.s_grid_u[i][j]['t'][:]
        tl = loopObj.s_grid_l[i][j]['t'][:]
        #value arrays
        sui = np.abs(loopObj.s_grid_u[i][j]["y"][0, :])
        sli = np.abs(loopObj.s_grid_l[i][j]["y"][0, :])
        suo = np.abs(loopObj.s_grid_0u[i][j]["y"][0, :])
        slo = np.abs(loopObj.s_grid_0l[i][j]["y"][0, :])
        su2= np.abs(loopObj.psij(tu))
        sl2= np.abs(loopObj.psij(tl))


        #--PLOTTING--
        fig, ax = plt.subplots()
        ax.set_ylabel('Absolute value of \u03A8')
        ax.set_xlabel('x (m)')
        plt.title(f"\u03A8 envelope, dk={k}, R={R}, B={B}, \u03BC={mu}")
        line1, = ax.plot(tu, sui, color = 'red', label = 'with e-e interaction')
        line2, = ax.plot(tu, suo, color = 'green', label = 'without e-e interaction')
        line3, = ax.plot(tu, su2, color = 'blue', label = 'exact solution')
        line4, = ax.plot(tl, sli, color = 'red', label = 'with e-e interaction')
        line5, = ax.plot(tl, slo, color = 'green', label = 'without e-e interaction')
        line6, = ax.plot(tl, sl2, color = 'blue', label = 'exact solution')
        ax.legend(handles=[line1, line2, line3])
        ax.set_box_aspect(2.0/3.5)
        
        #plt.show()
        plt.savefig(f"plot_{i}_{j}_abs.pdf", format='pdf')
        plt.close()




