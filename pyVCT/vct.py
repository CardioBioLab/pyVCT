import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pyVCT.hamiltonian import calcdH, MAX_FOCALS
from pyVCT.utils import parse_config

class VCT:
    def __init__(
        self,
        tissue_type: str = "neonatal_rat",
        scenario: str = "monolayer_on_fiber",
        sizeX: float = 1.0,
        sizeY: float = 1.0,
        ncx: int = 10,
        ncy: int = 10,
        part: float = 0.5,
        fibers = False,
        seed: int = 42,
        path2params: str = None,
        voxsize: float = 0.0025,
        fibers_dist: float = 0.1,
    ) -> None:
        """
        Initiliaze Virual Cardiac Monolayer class

            Params:
                tissue_type: str, type of tissue, by deafault provided 'neonatal_rat'
                scenario: str, one of possible simulation modes. One of 'monolayer_on_fiber', 'monolayer_without_fiber', 'single_without_fiber'
                sizeX: int, x size of simulation area [mm]
                sizeY: int, y size of simualtion area [mm]
                ncx: int, number of cells by x axis
                ncy: int, number of by y axis
                part: float, % of fibroblasts in monolayear between 0 and 1
                seed: int, random seed for numpy generators
                energy_config: dict, config with energy params
                voxsize: float, size of voxels [mm]
                fibers_dist: float, distance between fibers [mm] 

            Returns:
                None
        """

        # fixate random state
        np.random.seed(seed)

        # cell and energy params
        if not path2params:
            path2params = os.path.dirname(__file__) + "/config" + "/params.yaml"

        self.energy_config = parse_config(path2params, tissue_type, scenario)

        # get size in voxels
        self.voxsize = voxsize
        self.sizeX, self.sizeY = sizeX, sizeY

        # init ncx and ncy
        self.ncx = ncx
        self.ncy = ncy

        # init partition
        self.part = part

        self.moves = 0

        self._init_constants()
        self._cast_units()
        self._init_cells()
        self._init_fibers(fibers)

    def _init_constants(self):
        """
        Initialize size constants for simulation
        """

        # define initial  size of cell
        self.startvolume = self.energy_config["TARGETVOLUME_FB"] / 10
        self.r = int(np.sqrt(self.startvolume) / 2)

        # define size of area
        self.marginx, self.marginy = int(self.sizeX / 10 / self.voxsize), int(self.sizeY / 10 / self.voxsize)
        self.nvx = int(self.sizeX / self.voxsize) + self.marginx
        self.nvy = int(self.sizeY / self.voxsize) + self.marginy
        
        # constans for fibers
        self.fibers_distance = int(0.1 / self.voxsize)
        
               
    def _cast_units(self):
        scale = self.voxsize / 0.0025
        self.energy_config['NUCLEI_R'] =  self.energy_config['NUCLEI_R'] / self.voxsize
        self.energy_config['GN_CM'] = self.energy_config['GN_CM'] / scale
        self.energy_config['GN_FB'] = self.energy_config['GN_FB'] / scale
        self.energy_config['TARGETVOLUME_CM'] = self.energy_config['TARGETVOLUME_CM'] / 1000 / self.voxsize / self.voxsize
        self.energy_config['TARGETVOLUME_FB'] = self.energy_config['TARGETVOLUME_FB'] / 1000 / self.voxsize / self.voxsize
        self.energy_config['INELASTICITY_CM'] = self.energy_config['INELASTICITY_CM'] * (scale**4)
        self.energy_config['INELASTICITY_FB'] = self.energy_config['INELASTICITY_FB'] * (scale**4)
        self.energy_config['JCMMD'] = self.energy_config['JCMMD'] * self.voxsize
        self.energy_config['JFBMD'] = self.energy_config['JFBMD'] * self.voxsize
        self.energy_config['JCMCM'] = self.energy_config['JCMCM'] * self.voxsize
        self.energy_config['JFBFB'] = self.energy_config['JFBFB'] * self.voxsize
        self.energy_config['JFBCM'] = self.energy_config['JFBCM'] * self.voxsize
        self.energy_config['LMAX_CM'] = self.energy_config['LMAX_CM'] / 1000 / self.voxsize
        self.energy_config['LMAX_FB'] = self.energy_config['LMAX_FB'] / 1000 / self.voxsize
        


    def _init_cells(self):

        # create arrays for simulation
        # cell tag at every voxel
        self.ctags = np.zeros((self.nvx, self.nvy), dtype=int) 
        # mark if there is contact with surface at voxel
        self.contacts = np.zeros((self.nvx, self.nvy))
        
        # cell type for every cell tag
        self.types = np.zeros((self.ncx * self.ncy + 1), dtype=int)
        # cell size for every cell tag
        self.cell_sizes = np.ones((self.ncx * self.ncy + 1), dtype=int) * (2 * self.r + 1)**2
        # number contacts with surface for every cell tag
        self.attached = np.zeros((self.ncx * self.ncy + 1), dtype=int)
        # mass centers for every cell tag
        self.mass_centers = np.ones((self.ncx * self.ncy + 1, 2))
        
        self.bonds = np.zeros((self.nvx, self.nvy, 2))

        dx = (self.nvx - 2 * self.marginx) / self.ncx
        dy = (self.nvy - 2 * self.marginy) / self.ncy

        assert dx > 2 * self.r and dy > 2 * self.r, "Too dense, please decrease number of cells"

        cell_number = 0
        for i in range(self.ncx):
            for j in range(self.ncy):

                dvx = np.random.randint(0, 2**32 - 1) % int(dx - 2 * self.r + 1) - (dx / 2 - self.r)
                dvy = np.random.randint(0, 2**32 - 1) % int(dx - 2 * self.r + 1) - (dx / 2 - self.r)
                x = self.marginx + int((i + 0.5) * dx + dvx)
                y = self.marginy + int((j + 0.5) * dx + dvy)

                cell_number += 1

                type = np.random.binomial(1, self.part) + 1

                self.ctags[x - self.r : x + self.r + 1, y - self.r : y + self.r + 1] = cell_number
                self.types[cell_number] = type
                self.mass_centers[cell_number] = [x, y]

    def _init_fibers(self, fibers):
        self.fibers = np.zeros((self.nvx, self.nvy))
        if not fibers:
            return 
        else:
            for i in range(self.fibers_distance, self.nvx, self.fibers_distance):
                self.fibers[i] = 1

        
    def step(self):
        """
        Makes 1 simulation step
        """
        for _ in range(self.nvx * self.nvy):
            
            # pick random voxel
            x_target = (
                np.random.randint(0, 2**32 - 1) % (self.nvx - self.marginx)
                + self.marginx // 2
            )
            y_target = (
                np.random.randint(0, 2**32 - 1) % (self.nvy - self.marginy)
                + self.marginy // 2
            )

            # pick random neigbour
            x_source = x_target + np.random.randint(0, 2**32 - 1) % 3 - 1
            y_source = y_target + np.random.randint(0, 2**32 - 1) % 3 - 1
            
            
            ttag = self.ctags[x_target, y_target]
            stag = self.ctags[x_source, y_source]

            #  if source and target id are the same -> skip step
            if ttag == stag:
                continue
            
            # if cell volume may become zero -> skip step
            if self.cell_sizes[ttag] <= 1:
                continue
            
            #if there was no bond before and it is needed create it
            if self.types[ttag] * self.types[stag] == 1 \
                and self.contacts[ttag] * self.contacts[stag] == 1 \
                and self.bonds[x_target, y_target].sum()!= 0 \
                and self.bonds[x_source, y_source].sum()!= 0:
                    self.bonds[x_target, y_target] = [x_source, y_source]
                    self.bonds[x_source, y_source] = [x_target, y_target]
                    
            dH = calcdH(
                self.ctags,
                self.types,
                self.fibers,
                self.contacts,
                self.mass_centers,
                self.bonds,
                self.cell_sizes,
                x_target,
                y_target,
                x_source,
                y_source,
                self.energy_config,
            )

            proba = np.exp(-dH) if dH > 0 else 1
            
            if np.random.binomial(1, proba) > 0:
                self.move(x_target, y_target, x_source, y_source)

    def move(self, x_target, y_target, x_source, y_source):
        """
        Move source voxel to target voxel
        """
        self.moves += 1

        ttag = self.ctags[x_target, y_target]
        stag = self.ctags[x_source, y_source]

        # update cell tags
        self.ctags[x_target, y_target] = stag
        
        # update mass centers and cell sizes
        if ttag != 0:
            self.cell_sizes[ttag] -= 1
            self.update_mass_centers(ttag)
           
        if stag != 0:
            self.cell_sizes[stag] += 1
            self.update_mass_centers(stag)
            
        # update contacts and attached
        # move contact value from source to target
        if self.contacts[x_target, y_target]:
            self.attached[ttag] -= 1
        self.contacts[x_target, y_target] = self.contacts[x_source, y_source]
        self.contacts[x_source, y_source] = 0

        # if there are few contact in source cell add new contact to occupied voxel
        if stag and (not self.contacts[x_target, y_target]) and self.attached[stag] < self.energy_config[MAX_FOCALS(self.types[stag])]:
            self.contacts[x_target, y_target] = 1
            self.attached[stag] += 1 
        
        # Update bonds  
        if self.bonds[x_source, y_source].sum() != 0:
            self.bonds[self.bonds[x_source, y_source]] = [0, 0]
            self.bonds[x_source, y_source] = [0, 0]
        
        if self.bonds[x_target, y_target].sum() != 0:
            self.bonds[self.bonds[x_target, y_target]] = [0, 0]
            self.bonds[x_target, y_target] = [0, 0]
        
 
    def update_mass_centers(self, tag):
        '''
        Update mass center for one cell
        '''
        
        idxs = np.argwhere(self.ctags == tag)
        x_center = idxs[:, 0].mean()
        y_center = idxs[:, 1].mean()
        self.mass_centers[tag] = [x_center, y_center]
        
        return x_center, y_center
        

    def draw(self):
        isCM = np.vectorize(lambda x: self.types[x] == 1)
        isFB = np.vectorize(lambda x: self.types[x] == 2)

        CMs = isCM(self.ctags)
        FBs = isFB(self.ctags)

        img = np.zeros(CMs.shape + (3,))
        img[:, :, 0] = CMs * 255
        img[:, :, 2] = FBs * 255
        img[:, :, 1] = self.fibers * 255

        plt.imshow(img)


if __name__ == "__main__":
    vct = VCT()
    vct.step()
