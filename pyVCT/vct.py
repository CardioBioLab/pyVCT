import numpy as np




class VCT:
    def __init__(
        self,
        sizeX: float = 1.0,
        sizeY: float = 1.0,
        ncx: int = 5,
        ncy: int = 5,
        part: float = 0.5,
        energy_config: dict={'TARGETVOLUME_FB' : 56},
        voxsize=0.0025,
    ) -> None:
        """
        Initiliaze Virual Cardiac Monolayer class 

            Params:
                sizeX: int, x size of simulation area [mm]
                sizeY: int, y size of simualtion area [mm]
                ncx: int, number of cells by x axis 
                ncy: int, number of by y axis
                part: float, % of fibroblasts in monolayear between 0 and 1
                energy_config: dict, config with energy params
                voxsize:  size of voxels [mm]

            Returns: 
                None                
        """

        # get size in voxels
        self.voxsize = voxsize
        self.sizeX, self.sizeY = sizeX, sizeY
        
        # cell and energy params 
        self.energy_config = energy_config

        # init ncx and ncy
        self.ncx = ncx
        self.ncy = ncy

        self.part = part

        self._init_constants()
        self._init_cells()
        
    def _init_constants(self):  
        
        # define initial  size of cell
        self.startvolume = self.energy_config['TARGETVOLUME_FB']  / 10
        self.r = int(np.sqrt(self.startvolume) / 2)
        
        # define size of area 
        self.marginx, self.marginy = int(self.sizeX / 10 / self.voxsize), int(self.sizeY / 10 / self.voxsize)
        self.nvx = int(self.sizeX / self.voxsize) + self.marginx
        self.nvy = int(self.sizeY / self.voxsize) + self.marginy
        

    def _init_cells(self):
        
         # create arrays for simulation
        self.ctags = np.zeros((self.nvx, self.nvy))
        self.types = np.zeros((self.nvx, self.nvy))
        self.contacts = np.zeros((self.nvx, self.nvy))
        self.bonds = np.zeros((self.nvx, self.nvy))

        dx = (self.nvx - 2 * self.marginx) / self.ncx 
        dy = (self.nvy - 2 * self.marginy) / self.ncy
        
        #cell radius
       

        assert dx > 2 * self.r and dy > 2 * self.r, "Too dense"
    
        cell_num = 0
        for i in range(self.ncx):
            for j in range(self.ncy):
                
                dvx = np.random.randint(0, 2**32-1) % int(dx-2*self.r+1) -(dx/2 - self.r)
                dvy = np.random.randint(0, 2**32-1) % int(dx-2*self.r+1) -(dx/2 - self.r)
                x = self.marginx + int((i + 0.5) * dx + dvx)
                y = self.marginy + int((j + 0.5) * dx + dvy)
                
                cell_num += 1   
                
                type = np.random.binomial(1, self.part) + 1
                
                self.ctags[x-self.r:x+self.r , y-self.r:y+self.r] = cell_num
                self.types[cell_num] = type
        
        
if __name__ == "__main__":
    vct = VCT(0.1, 0.1)
