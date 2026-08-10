"""
Ionic model configuration for a four-chamber simulation.

Vendored verbatim from SIMULATION_library.fch_cell by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

class ionic:
	def __init__(self,
				 impID,
				 name,
				 ionic_model,
				 IDs_list,
				 param=None,
				 plugin=None,
				 plug_param=None,
				 im_sv_init=None				
			 ):

		self.impID = impID
		self.name = name
		self.ionic_model = ionic_model
		self.IDs_list = IDs_list
		self.param = param
		self.plugin = plugin
		self.plug_param = plug_param
		self.im_sv_init = im_sv_init

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('Ionic model @ '+self.name+':')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')
		
	def write2file(self,
				   fileID):

		fileID.write('  -imp_region['+str(self.impID)+'].name '+self.name+' \\\n')
		fileID.write('  -imp_region['+str(self.impID)+'].im '+self.ionic_model+' \\\n')
		fileID.write('  -imp_region['+str(self.impID)+'].num_IDs '+str(len(self.IDs_list))+' \\\n')

		for i,idx in enumerate(self.IDs_list):
			fileID.write('  -imp_region['+str(self.impID)+'].ID['+str(i)+'] '+str(idx)+' \\\n')
		if self.param is not None:
			fileID.write('  -imp_region['+str(self.impID)+'].im_param '+self.param+' \\\n')

		if self.ionic_model!='PASSIVE':
			fileID.write('  -imp_region['+str(self.impID)+'].im_sv_dumps Ca_i \\\n')

		if self.plugin is not None:
			fileID.write('  -imp_region['+str(self.impID)+'].plugins '+self.plugin+' \\\n')
		if self.plug_param is not None:
			fileID.write('  -imp_region['+str(self.impID)+'].plug_param '+self.plug_param+' \\\n')
		if self.im_sv_init is not None:
			fileID.write('  -imp_region['+str(self.impID)+'].im_sv_init '+self.im_sv_init+' \\\n')
