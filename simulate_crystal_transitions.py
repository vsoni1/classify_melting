import numpy as np
import hoomd
import hoomd.md
import hoomd.dem
import hoomd.md.force as force

def run_sim(index,temperature,path):
    """Runs simulation of interacting dipoles at given temperature 
    and saves the result 

    Parameters
    ----------
    index : tag to append to file name
    temperature : temperature of simulation
    path : path to save to
    """
    hoomd.context.initialize("--gpu=1")
    positions = np.load('positions_small.npy')
    snapshot = hoomd.data.make_snapshot(
        N=625, box=hoomd.data.boxdim(Lx=12.5,Ly=12.5,dimensions=2),
        particle_types=['A']
        );
    snapshot.particles.position[:] = positions
    snapshot.particles.mass[:] = 2
    system = hoomd.init.read_snapshot(snapshot);

    def interaction(r, rmin, rmax, epsilon):
        V = epsilon*(-(1/r)**3);
        F = epsilon / r * (- 3 * (1/ r)**3);
        return (V, F)

    all = hoomd.group.all();
    nl = hoomd.md.nlist.cell();
    groupA = hoomd.group.type('A')

    table = hoomd.md.pair.table(width=1000,nlist=nl)
    table.pair_coeff.set(
        'A', 'A', func=interaction, rmin=0.01, rmax=2,
        coeff=dict(epsilon=-100)
        )
    
    langevin = hoomd.md.integrate.langevin(
        group=all, kT=temperature, seed=42+i
        );
    langevin.set_gamma('A',gamma=100)
    hoomd.md.integrate.mode_standard(dt=.001);
    hoomd.run(1000);
    langevin.set_gamma('A',gamma=1)
    hoomd.md.integrate.mode_standard(dt=.0001);
    hoomd.run(10000);
    hoomd.dump.gsd(
        path+"/lj_fluid3_unforced_kT"+str(temperature)+"_"+str(index)+".gsd", 
        period=100, group=all, overwrite=True
        );
    hoomd.run(500);