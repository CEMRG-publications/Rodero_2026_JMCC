"""
Electrophysiology regions, stimuli, and the fast conduction clock.

Vendored verbatim from SIMULATION_library.fch_electrophysiology by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

class fcclock:
	def __init__(self,
				 BCL,
				 SA_t0,
				 AV_delay,
				 ERP=600.0,
				 SA_RA_delay=0.0,
				 AA_delay=0.0,
				 VV_delay=0.0,
				 ):

		self.BCL = BCL
		self.SA_t0 = SA_t0
		self.AV_delay = AV_delay

		# if LA_LV_delay is None:
		# 	if VV_delay>=0:
		# 		self.LA_LV_delay = AV_delay
		# 	else:
		# 		self.LA_LV_delay = AV_delay + abs(VV_delay)

		# if RA_RV_delay is None:
		# 	if VV_delay>=0: # means LV ahead
		# 		self.RA_RV_delay = AV_delay + abs(VV_delay)
		# 	else:
		# 		self.RA_RV_delay = AV_delay 

		self.ERP = ERP
		self.SA_RA_delay = SA_RA_delay
		self.AA_delay = AA_delay
		self.VV_delay =  VV_delay

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('Four-chamber clock:')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')
		
	def write2file(self,
				   fileID):

		fileID.write('  -fcclock.SA_CL '+str(self.BCL)+' \\\n')
		fileID.write('  -fcclock.ERP '+str(self.ERP)+' \\\n')
		fileID.write('  -fcclock.SA_t0 '+str(self.SA_t0)+' \\\n')
		fileID.write('  -fcclock.SA_RA_delay '+str(self.SA_RA_delay)+' \\\n')
		fileID.write('  -fcclock.AA_delay '+str(self.AA_delay)+' \\\n')
		fileID.write('  -fcclock.VV_delay '+str(self.VV_delay)+' \\\n')
		# fileID.write('  -fcclock.RA_AVa_delay '+str(self.RA_AVa_delay)+' \\\n') # these are set automatically by CARP 
		fileID.write('  -fcclock.AV_delay '+str(self.AV_delay)+' \\\n')
		# fileID.write('  -fcclock.LA_LV_delay '+str(self.LA_LV_delay)+' \\\n')
		# fileID.write('  -fcclock.RA_RV_delay '+str(self.RA_RV_delay)+' \\\n')

class stimulus:
	def __init__(self,
				 stimID,
				 name,
				 BCL,
				 npls,
				 xtrg,
				 stim_type,
				 vtx_file=None,
				 start=0.0,
				 strength=60.0,
				 duration=2.0,
				 xtrg_offset=0.0):

		self.stimID = stimID
		self.name = name
		self.vtx_file = vtx_file
		self.BCL = BCL
		self.npls = npls
		self.xtrg = xtrg
		self.stim_type = stim_type
		self.start = start
		self.strength = strength
		self.duration = duration
		self.xtrg_offset = xtrg_offset

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('Stimulus @ '+self.name+':')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')
		
	def write2file(self,
				   fileID):
		if self.stim_type!=8:
			fileID.write('  -stimulus['+str(self.stimID)+'].name '+self.name+' \\\n')
			fileID.write('  -stimulus['+str(self.stimID)+'].vtx_file '+self.vtx_file+' \\\n')
			fileID.write('  -stimulus['+str(self.stimID)+'].start '+str(self.start)+' \\\n')
			fileID.write('  -stimulus['+str(self.stimID)+'].strength '+str(self.strength)+' \\\n')
			fileID.write('  -stimulus['+str(self.stimID)+'].xtrg_offset '+str(self.xtrg_offset)+' \\\n')
			fileID.write('  -stimulus['+str(self.stimID)+'].xtrg '+str(self.xtrg)+' \\\n')
			
		fileID.write('  -stimulus['+str(self.stimID)+'].stimtype '+str(self.stim_type)+' \\\n')
		fileID.write('  -stimulus['+str(self.stimID)+'].npls '+str(self.npls)+' \\\n')
		fileID.write('  -stimulus['+str(self.stimID)+'].bcl '+str(self.BCL)+' \\\n')
		fileID.write('  -stimulus['+str(self.stimID)+'].duration '+str(self.duration)+' \\\n')

class ep_region:
	def __init__(self,
				 name,
				 gregionID,
				 IDs_list,
				 CV_list,
				 ignore=False
				 ):

		self.name = name
		self.gregionID = gregionID
		self.IDs_list = IDs_list
		self.CV_list = CV_list
		self.ignore = ignore

	def visualise(self):

		dct = self.__dict__

		print('----------------------------')
		print('EP region @ '+self.name+':')
		for k in dct:
			print(k+' : '+str(dct[k]))		
		print('----------------------------')
		
	def write2file(self,
				   fileID):		

		fileID.write('  -gregion['+str(self.gregionID)+'].name '+self.name+' \\\n')
		fileID.write('  -gregion['+str(self.gregionID)+'].num_IDs '+str(len(self.IDs_list))+' \\\n')
		for i,idx in enumerate(self.IDs_list):
			fileID.write('  -gregion['+str(self.gregionID)+'].ID['+str(i)+'] '+str(idx)+' \\\n')
		fileID.write('  -ekregion['+str(self.gregionID)+'].ID '+str(self.gregionID)+' \\\n')
		if not self.ignore:
			fileID.write('  -ekregion['+str(self.gregionID)+'].vel_f '+str(self.CV_list[0])+' \\\n')
			fileID.write('  -ekregion['+str(self.gregionID)+'].vel_s '+str(self.CV_list[1])+' \\\n')
			fileID.write('  -ekregion['+str(self.gregionID)+'].vel_n '+str(self.CV_list[2])+' \\\n')
		else:
			fileID.write('  -ekregion['+str(self.gregionID)+'].ignore 1 \\\n')
