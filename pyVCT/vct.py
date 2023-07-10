import os

import numpy as np

from pyVCT.utils import parse_config
from pyVCT.hamiltonian import calcdH


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
        seed: int = 42,
        path2params: str = None,
        voxsize=0.0025,
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
                voxsize:  size of voxels [mm]

            Returns:
                None
        """

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

        self.part = part

        self._init_constants()
        self._init_cells()

    def _init_constants(self):
        '''
        Initialize size constants for simulation
        '''

        # define initial  size of cell
        self.startvolume = self.energy_config["TARGETVOLUME_FB"] / 10
        self.r = int(np.sqrt(self.startvolume) / 2)

        # define size of area
        self.marginx, self.marginy = int(self.sizeX / 10 / self.voxsize), int(self.sizeY / 10 / self.voxsize)
        self.nvx = int(self.sizeX / self.voxsize) + self.marginx
        self.nvy = int(self.sizeY / self.voxsize) + self.marginy

    def _init_cells(self):

        # create arrays for simulation
        self.ctags = np.zeros((self.nvx, self.nvy), dtype=int)
        self.types = np.zeros((self.ncx * self.ncy + 1), dtype=int)
        self.contacts = np.zeros((self.nvx, self.nvy))
        self.bonds = np.zeros((self.nvx, self.nvy))
        self.mass_centers = np.zeros((self.ncx * self.ncy + 1, 2))

        dx = (self.nvx - 2 * self.marginx) / self.ncx
        dy = (self.nvy - 2 * self.marginy) / self.ncy


        assert dx > 2 * self.r and dy > 2 * self.r, "Too dense"

        cell_number = 0
        for i in range(self.ncx):
            for j in range(self.ncy):

                dvx = np.random.randint(0, 2**32 - 1) % int(dx - 2 * self.r + 1) - (dx / 2 - self.r)
                dvy = np.random.randint(0, 2**32 - 1) % int(dx - 2 * self.r + 1) - (dx / 2 - self.r)
                x = self.marginx + int((i + 0.5) * dx + dvx)
                y = self.marginy + int((j + 0.5) * dx + dvy)

                cell_number += 1

                type = np.random.binomial(1, self.part) + 1

                self.ctags[x - self.r : x + self.r, y - self.r : y + self.r] = cell_number
                self.types[cell_number] = type
                self.mass_centers[cell_number] = [x, y]
                
    def step(self):
        '''
        '''
        
        # pick random grid element
        x = np.random.randint(0, 2**32 - 1) % self.nvx
        y = np.random.randint(0, 2**32 - 1) % self.nvx
        
        x_niegbour = np.random.randint(0, 2**32 - 1) % 8
        y_niegbour = np.random.randint(0, 2**32 - 1) % 8
        
         
        return
        
        


if __name__ == "__main__":
    vct = VCT()
