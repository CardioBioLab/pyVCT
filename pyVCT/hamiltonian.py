'''
ctags: 2d array. just cell id of each pixel.
CMs: [[x1,y1],[x2,y2],[x3,y3]...[xi,yi]] where i is a cell tag 
bond: 2d array each element is [xi,yi]
types: 1d array. !first element is always 0, other are cell types with correspond to their id in ctags

params: dictionary with following energy parameters
    TARGETVOLUME_CM, TARGETVOLUME_FB,
    INELASTICITY_CM, INELASTICITY_FB,
    GN_CM, GN_FB,
    DETACH_CM, DETACH_FB,
    UNLEASH_CM, UNLEASH_FB,
    LMAX_CM, LMAX_FB,
    JCMCM, JFBCM, JFBFB, JCMMD, JFBMD,
    JH,
    JB,
    G_NCH,
    E_bond,
    F_ANGLE,
    NUCLEI_R,
    NUCL
'''
import numpy as np

def TARGETVOLUME(a):
    if a == 1:
        return 'TARGETVOLUME_CM'
    else:
        return 'TARGETVOLUME_FB'
    
def INELASTICITY(a):
    if a == 1:
        return 'INELASTICITY_CM'
    else:
        return 'INELASTICITY_FB'
    
def GN(a):
    if a == 1:
        return 'GN_CM'
    else:
        return 'GN_FB'
    
def DETACH(a):
    if a == 1:
        return 'DETACH_CM'
    else:
        return 'DETACH_FB'
    
def UNLEASH(a):
    if a == 1:
        return 'UNLEASH_CM'
    else:
        return 'UNLEASH_FB'

def LMAX(a):
    if a == 1:
        return 'LMAX_CM'
    else:
        return 'LMAX_FB'


def calcdH_CH(ctags, contacts, CMs, xt, yt, xs, ys, params):
    """
    Calculate dH for channel distribution

	    Params:
		    ctags: numpy 2D array, ctags[i][j] is id of cell with coordinates x=i and y=j
		    contacts: numpy 2D array, contacts[i][j] can be 0 or 1 and show is there a contact in voxel with x=i and y=j
		    CMs: numpy 2D array, CMs[i]=[xi,yi] contains coordinates of center of mass for cell with id=i
		    xt: int, x coordinate of target voxel
		    yt: int, y coordinate of target voxel
		    xs: int, x coordinate of source voxel
		    ys: int, y coordinate of source voxel
		    params: dict, contains all energy constants

	    Returns:
		    dH: float, difference in energy 
    """
    dHborder = 0
    dHborder = calcdHborder(ctags, contacts, xt, yt, params)

    dHdist = 0
    dHdist = calcdHdist(ctags, contacts, CMs, xt, yt, xs, ys, params)

    dH = dHdist + dHborder

    return dH

def calcdHborder(ctags, contacts, xt, yt, params):
    ttag = ctags[xt][yt]
    dHcontact = params['JB']
    nbs = [[xt,yt+1],[xt,yt-1],[xt+1,yt+1],[xt+1,yt],[xt+1,yt-1],[xt-1,yt+1],[xt-1,yt],[xt-1,yt-1]] 
    for n in range(8):
        nbtag = ctags[nbs[n][0]][nbs[n][1]]
        if ttag != nbtag:
            if ((contacts[nbs[n][0]][nbs[n][1]]) and (nbtag != 0)):
                dHcontact = 0
            else:
                dHcontact = params['JH']
            break
    return dHcontact

def calcdHdist(ctags, contacts, CMs, xt, yt, xs, ys, params):
    dH = 0

    if contacts[xs][ys]:
        dH = params['G_NCH'] / dist(CMs, xt, yt, ctags[xt][yt])

    return dH

def calcdH(ctags, types, fibers, contacts, CMs, bond, csize, xt, yt, xs, ys, params):
    """
    Calculate dH

	    Params:
		    ctags: numpy 2D array, ctags[i][j] is id of cell with coordinates x=i and y=j, ctags.shape = (NVX,NVY)
		    types: numpy 1D array, types[i] is a type of cell with id=i it can be 1 or 2, but types[0]=0 for non-cell type(medium), len(types)=(Number_of_cells+1)
		    fibers: numpy 2D array, fibers[i][j] can be 0 or 1 and show is there a fiber in voxel with x=i and y=j, fibers.shape = (NVX,NVY)
		    contacts: numpy 2D array, contacts[i][j] can be 0 or 1 and show is there a contact in voxel with x=i and y=j, contacts.shape = (NVX,NVY)
		    CMs: numpy 2D array, CMs[i]=[xi,yi] contains coordinates of center of mass for cell with id=i, len(CMs)=(Number_of_cells+1)
		    bond: numpy 3D array, bond[i][j]=[x,y] shows that voxel with coordinates i,j has connection with voxel at x,y place, bond.shape = (NVX,NVY,2)
		    csize: numpy 1D array, csize[i] contains size of cell with id=i, len(csize)=(Number_of_cells+1)
		    xt: int, x coordinate of target voxel
		    yt: int, y coordinate of target voxel
		    xs: int, x coordinate of source voxel
		    ys: int, y coordinate of source voxel
		    params: dict, dictionary with following energy parameters
    				TARGETVOLUME_CM, TARGETVOLUME_FB,
    				INELASTICITY_CM, INELASTICITY_FB,
    				GN_CM, GN_FB,
    				DETACH_CM, DETACH_FB,
    				UNLEASH_CM, UNLEASH_FB,
    				LMAX_CM, LMAX_FB,
    				JCMCM, JFBCM, JFBFB, JCMMD, JFBMD,
    				JH,
    				JB,
    				G_NCH,
    				E_bond,
    				F_ANGLE,
    				NUCLEI_R,
    				NUCL

	    Returns:
		    dH: float, difference in energy 
    """
    
    dH = 0
    dHcontact = 0
    dHvol = 0
    # dHfocals = 0
    # dHsyncytium = 0
    # dHnuclei = 0
    
    
    dHcontact = calcdHcontact(ctags, types, xt, yt, xs, ys, params)
    
    dHvol = calcdHvol(csize, ctags[xt][yt], ctags[xs][ys], types[ctags[xt][yt]], types[ctags[xs][ys]], params)
    
    # dHfocals = calcdHprotrude(ctags, types, contacts, CMs, xt, yt, xs, ys, fibers[xt][yt], fibers[xs][ys], params)
    
    # if params['E_bond']:
    #     dHsyncytium = calcdHsyncytium(ctags, CMs, bond, xt, yt, xs, ys, params)
    
    # dHnuclei = calcdHnuclei(ctags, types, CMs, xt, yt, params)
    
    dH = dHcontact + dHvol #+ dHfocals + dHsyncytium + dHnuclei
    
    return dH

def calcdHsyncytium(ctags, CMs, bond, xt, yt, xs, ys, params):
    dH=0
    if bond[xs][ys][0] != 0 and bond[xs][ys][1] != 0:
        xsy = ys - CMs[ctags[xs][ys]][1]
        xsx = xs - CMs[ctags[xs][ys]][0]

        
        btag = ctags[bond[xs][ys][0]][bond[xs][ys][1]]
        xby = bond[xs][ys][1] - CMs[btag][1]
        xbx = bond[xs][ys][0] - CMs[btag][0]

        normX = np.sqrt(xsy * xsy + xsx * xsx)
        normB = np.sqrt(xby * xby + xbx * xbx)

        vx = xsx / normX + xbx / normB
        vy = xsy / normX + xby / normB

        dH += params['E_bond'] * (2 - np.sqrt(vx * vx + vy * vy))

    if bond[xt][yt][0] != 0 and bond[xt][yt][1] != 0:
        xsy = yt - CMs[ctags[xt][yt]][1]
        xsx = xt - CMs[ctags[xt][yt]][0]

        btag = ctags[bond[xt][yt][0]][bond[xt][yt][1]]
        xby = bond[xt][yt][1] - CMs[btag][1]
        xbx = bond[xt][yt][0] - CMs[btag][0]


        normX = np.sqrt(xsy * xsy + xsx * xsx)
        normB = np.sqrt(xby * xby + xbx * xbx)

        vx = xsx / normX + xbx / normB
        vy = xsy / normX + xby / normB

        dH += params['E_bond'] * (2 - np.sqrt(vx * vx + vy * vy))

    return dH

def calcdHcontact(ctags, types, xt, yt, xs, ys, params):
    dHcontact = 0
    Hcontact = 0
    Hcontactn = 0
    nbs = [[xt,yt+1],[xt,yt-1],[xt+1,yt+1],[xt+1,yt],[xt+1,yt-1],[xt-1,yt+1],[xt-1,yt],[xt-1,yt-1]]
    
    for n in range(8):
        nbtag = ctags[nbs[n][0]][nbs[n][1]]
        Hcontact += contactenergy(ctags[xt][yt], nbtag, types[ctags[xt][yt]], types[nbtag],params)
        Hcontactn += contactenergy(ctags[xs][ys], nbtag, types[ctags[xs][ys]], types[nbtag],params)
    
    dHcontact = Hcontactn - Hcontact
    
    return dHcontact

def contactenergy(tag1, tag2, type1, type2, params):
    J = 0
    if tag1 != tag2:
        if tag1 == 0 or tag2 == 0:
            type = type2 if tag1 == 0 else type1
            J = params['JCMMD'] if type == 1 else params['JFBMD']
        elif type1 == type2:
            J = params['JCMCM'] if type1 == 1 else params['JFBFB']
        else:
            J = params['JFBCM']

    return J

def calcdHvol(csize, ttag, stag, ttype, stype, params):
    dHvolA = 0
    dHvolB = 0
    if ttag:
        V0 = params[TARGETVOLUME(ttype)]
        V = csize[ttag-1]
        eV = (V-V0)/V0
        eVn = (V-1-V0)/V0
        dHvolA = params[INELASTICITY(ttype)]*(eVn*eVn-eV*eV)
    if stag:
        V0 = params[TARGETVOLUME(stype)]
        V = csize[stag-1]
        eV = (V-V0)/V0
        eVn = (V+1-V0)/V0
        dHvolB = params[INELASTICITY(stype)]*(eVn*eVn-eV*eV)
    dHvol = dHvolA + dHvolB

    return dHvol

def calcdHprotrude(ctags, types, contacts, CMs, xt, yt, xs, ys, Qt, Qs, params):
    dH = 0
    coss = 1
    cost = 1
    stag = ctags[xs][ys]
    ttag = ctags[xt][yt]

    if Qs and Qt:
        
        coss = np.cos(params['F_ANGLE'] - np.atan((ys - CMs[stag][1]) / (xs - CMs[stag][0])))
        
        
        cost = np.cos(params['F_ANGLE'] - np.atan((yt - CMs[ttag][1]) / (xt - CMs[stag][0])))

    if contacts[xs][ys]:
        distt = dist(CMs, xt, yt, ttag)
        dists = dist(CMs, xs, ys, stag)
        if (distt < params[LMAX(types[stag])]):
            if (dists < params[LMAX(types[stag])]):
                dH = params[GN(types[stag])] * ((1 / distt) * abs(1 / cost) - (1 / dists) * abs(1 / coss))
            else:
                dH = params[GN(types[stag])] * (1 / distt) * abs(1 / cost)
        else:
            dH = 9999999

        if contacts[xt][yt]:
            dH += params[DETACH(types[ttag])]

        if Qs and not Qt:
            dH += params[UNLEASH(types[stag])]
    else:
        if contacts[xt][yt]:
            dH = params[DETACH(types[ttag])]
        else:
            dH = 0

        if contacts[xt][yt] and types[ttag] == 0:
            print("Media contact!!!!")
    
    return dH

def calcdHnuclei(ctags, types, CMs, xt, yt, params):
    dH = 0
    ttag = ctags[xt][yt]
    if ttag and dist(CMs, xt, yt, ttag) < params['NUCLEI_R']:
        dH = params['NUCL'] * params[DETACH(types[ttag])]

    return dH

def dist(CMs, xt, yt, tag):
    
    return np.sqrt((xt - CMs[tag][0])**2 + (yt - CMs[tag][1])**2)