"""
Mechanics regions, springs, and Neumann boundary conditions.

Vendored verbatim from SIMULATION_library.fch_mechanics by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

class mregion:
	def __init__(self,
				 name,
				 mregID,
				 IDs_list,
				 mtype,
				 params
				 ):

		self.name = name
		self.mregID = mregID
		self.IDs_list = IDs_list
		self.mtype = mtype
		self.params = params

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('Mech region @ '+self.name+':')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')
		
	def write2file(self,
				   fileID):

		fileID.write('  -mregion['+str(self.mregID)+'].name '+self.name+' \\\n')
		fileID.write('  -mregion['+str(self.mregID)+'].type '+str(self.mtype)+' \\\n')
		fileID.write('  -mregion['+str(self.mregID)+'].num_IDs '+str(len(self.IDs_list))+' \\\n')
		for i,idx in enumerate(self.IDs_list):
			fileID.write('  -mregion['+str(self.mregID)+'].ID['+str(i)+'] '+str(idx)+' \\\n')
		fileID.write('  -mregion['+str(self.mregID)+'].params '+self.params+' \\\n')

class mech_nbc:
	def __init__(self,
				 name,
				 nbcID,
				 surf_file,
				 pressure=None,
				 spring=False,
				 spring_idx=None,
				 nspring_idx=None,
				 nspring_config=None,
				 trace_file=None):
		self.name = name
		self.nbcID = nbcID
		self.surf_file = surf_file
		self.pressure = pressure
		self.spring = spring
		self.spring_idx = spring_idx
		self.nspring_idx = nspring_idx
		self.nspring_config = nspring_config
		self.trace_file = trace_file

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('Neumann BC @ '+self.name+':')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')

	def write2file(self,
				   fileID):
		fileID.write('  -mechanic_nbc['+str(self.nbcID)+'].name '+self.name+' \\\n')
		fileID.write('  -mechanic_nbc['+str(self.nbcID)+'].surf_file '+self.surf_file+' \\\n')
		if self.spring:
			if (self.spring_idx is None) or (self.nspring_idx is None):
				raise Exception('You need to define the spring parameters for springs.')
			else:
				fileID.write('  -mechanic_nbc['+str(self.nbcID)+'].spring_idx '+str(self.spring_idx)+' \\\n')
				fileID.write('  -mechanic_nbc['+str(self.nbcID)+'].nspring_idx '+str(self.nspring_idx)+' \\\n')
				if (self.nspring_idx>=0) and (self.nspring_config is not None):
					fileID.write('  -mechanic_nbc['+str(self.nbcID)+'].nspring_config '+str(self.nspring_config)+' \\\n')
				elif (self.nspring_idx>=0) and (self.nspring_config is None):
					raise Exception('You need to define the spring configuration for normal springs.')
		else:
			if self.pressure is None:
				raise Exception('You need to provide a pressure value for Neumann BCs.')
			else:		
				fileID.write('  -mechanic_nbc['+str(self.nbcID)+'].pressure '+str(self.pressure)+' \\\n')
			if self.trace_file is not None:
				fileID.write('  -mechanic_nbc['+str(self.nbcID)+'].trace '+self.trace_file+' \\\n')

class spring:
	def __init__(self,
				 name,
				 springID,
				 stiffness,
				 ncomp=None,
				 elem_file=None):
		self.name = name
		self.springID = springID
		self.stiffness = stiffness
		self.ncomp = ncomp
		self.elem_file = elem_file

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('Spring BC @ '+self.name+':')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')

	def write2file(self,
				   fileID):

		fileID.write('  -mechanic_bs['+str(self.springID)+'].value '+str(self.stiffness)+' \\\n')
		if self.elem_file is not None:
			fileID.write('  -mechanic_bs['+str(self.springID)+'].edidx '+str(self.springID)+' \\\n')
			fileID.write('  -mechanic_ed['+str(self.springID)+'].ncomp '+str(self.ncomp)+' \\\n')
			fileID.write('  -mechanic_ed['+str(self.springID)+'].file '+self.elem_file+' \\\n')
