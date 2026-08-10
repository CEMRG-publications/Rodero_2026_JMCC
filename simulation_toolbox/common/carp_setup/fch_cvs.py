"""
Cavity definitions for the circulatory system model.

Vendored verbatim from SIMULATION_library.fch_cvs by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

class cavity:
	def __init__(self,
				 name,
				 cavID,
				 cav_type,
				 pID,
				 volID,
				 p0,
				 p0_in,
				 p0_out,
				 state=-1):
		self.name = name
		self.cavID = cavID
		self.cav_type = cav_type
		self.pID = pID
		self.volID = volID
		self.p0 = p0
		self.p0_in = p0_in
		self.p0_out = p0_out
		self.state = state

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('Cavity @ '+self.name+':')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')

	def write2file(self,
				   fileID):

		fileID.write('  -cavities['+str(self.cavID)+'].cav_type '+str(self.cav_type)+' \\\n')
		fileID.write('  -cavities['+str(self.cavID)+'].cavP '+str(self.pID)+' \\\n')
		fileID.write('  -cavities['+str(self.cavID)+'].cavVol '+str(self.volID)+' \\\n')
		fileID.write('  -cavities['+str(self.cavID)+'].p0_cav '+str(self.p0)+' \\\n')
		fileID.write('  -cavities['+str(self.cavID)+'].p0_in '+str(self.p0_in)+' \\\n')
		fileID.write('  -cavities['+str(self.cavID)+'].p0_out '+str(self.p0_out)+' \\\n')
		fileID.write('  -cavities['+str(self.cavID)+'].state '+str(self.state)+' \\\n')	