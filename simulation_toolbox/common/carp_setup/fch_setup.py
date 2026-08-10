"""
The simulation class holding a whole four-chamber setup.

Vendored verbatim from SIMULATION_library.fch_setup by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

import json

class simulation:
	def __init__(self,
				 meshdir=None,
				 meshname=None,
				 fibresname=None,
				 simulation_folder=None,
				 testname=None,
				 folderopts=None,
				 nproc=None,
				 walltime=None,
				 platform=None,
				 carpentryfolder=None,
				 nbeats=None,
				 initial_nbeats=None,
				 full_nbeats=None,
				 torord_init_file=None,
				 torord_rv_init_file=None,
				 courtemanche_init_file=None,
				 material_law=None,
				 loadStepping=40,
				 lvendo_name='LV_endo',
				 rvendo_name='RV_endo',
				 laendo_name='LA_endo',
				 raendo_name='RA_endo',
				 epi_name='epicardium',
				 peri_scale='pericardium_scale',
				 load_peri_on=None,
				 cycle_peri_on=None,
				 springs=None,
				 k_springs=None,
				 fast_newton=False,
				 stimuli=None,
				 clock_ID=None,
				 restart_state=None,
				 contraction_model='Land',
				 trace_file=None,
				 unload_atria=False,
				 mech_stiffness_damping=0.1
				 ):
		self.meshdir = meshdir
		self.meshname = meshname
		self.fibresname = fibresname
		self.testname = testname
		self.simulation_folder = simulation_folder
		self.folderopts = folderopts
		self.nproc = nproc
		self.walltime = walltime
		self.platform = platform
		self.carpentryfolder = carpentryfolder
		self.nbeats = nbeats
		self.initial_nbeats = initial_nbeats
		self.full_nbeats = full_nbeats
		self.torord_init_file = torord_init_file
		self.torord_rv_init_file = torord_init_file
		self.courtemanche_init_file = courtemanche_init_file
		self.material_law = material_law
		self.loadStepping = loadStepping
		self.lvendo_name = lvendo_name
		self.rvendo_name = rvendo_name
		self.laendo_name = laendo_name
		self.raendo_name = raendo_name
		self.epi_name = epi_name
		self.peri_scale = peri_scale
		self.load_peri_on = load_peri_on
		self.cycle_peri_on = cycle_peri_on
		self.springs = springs
		self.k_springs = k_springs
		self.fast_newton = fast_newton
		self.stimuli = stimuli
		self.clock_ID = clock_ID
		self.restart_state = restart_state
		self.contraction_model = contraction_model
		self.trace_file = trace_file
		self.unload_atria = unload_atria
		self.mech_stiffness_damping = mech_stiffness_damping
		
	def save(self,	
			 filename):

		print('Saving '+filename+'...')

		pat_dct = vars(self)

		obj_dct = {}
		obj_dct.update(
			{k: pat_dct[k] for k in set(list(pat_dct.keys()))}
		)

		with open(filename, 'w') as f:
		    json.dump(obj_dct, f, indent=4)
		f.close()

		print('Done.')

	def load(self,
			 filename):

		print('Loading '+filename+'...')

		f = open(filename,"r")
		obj_dct = json.load(f)
		f.close()

		for k, v in obj_dct.items():
			setattr(self, k, v)

		print('Done.')