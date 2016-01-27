// file: CM.c
#include "functions.h"

////////////////////////////////////////////////////////////////////////////////
CM* allocCM(int NRc)
// alloc CM
{
	CM* CMs;

	CMs = calloc(NRc+1, sizeof(CM));
	
	return CMs;
}

////////////////////////////////////////////////////////////////////////////////
CONT* allocContacts(int NRc)
// alloc Contacts
{
	CONT* cont;

	cont = calloc((NRc+1)*MAX_FOCALS, sizeof(CONT));
	
	return cont;
}

////////////////////////////////////////////////////////////////////////////////
void findCM(VOX* pv, CM* CMs, int NRc)
// find center of mass
{
	int vx,vy,v,i;
	long int x[NRc+1], y[NRc+1], n[NRc+1];

	for(i=0;i<=NRc;i++){
		x[i]=0;
		y[i]=0;
		n[i]=0;
	}

	for(vx=0; vx<NVX; vx++) {
        for (vy=0; vy<NVY; vy++) {
            v = vx + vy * NVX;
        	if (pv[v].ctag != 0 ){
				x[pv[v].ctag] += vx;
				y[pv[v].ctag] += vy;
				n[pv[v].ctag] ++;
			}
		}
    }

    for(i=0;i<=NRc;i++){
    	CMs[i].x = (double) x[i] / (double) n[i];
    	CMs[i].y = (double) y[i] / (double) n[i];
    }
}

////////////////////////////////////////////////////////////////////////////////