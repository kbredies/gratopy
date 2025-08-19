from __future__ import division, print_function

import numpy as np
import gratopy


class angle_information():
	"""
    Objects of the angle_information class contain three attributes,
    .Na 			... "the number of angles considered"
    .angles			... "the relevant angles"
    .angle_weights 	... "the corresponding angle weights"
    """
	def __init__(self,angles, angle_weights):
		"""
		Creates an angle_information instance given
		angles 			...	a list or numpy.array of angles
		angle_weights	... a list or numpy.array of angle_weights
		with corresponding attributes. The .Na attribute
		is set automatically after checking the length of angles
		and angle_weights coincide.
		"""
		assert(len(angles) == len(angle_weights)),(
			"Could not create angle_information."+
			"The length of the given angles (" + str(len(angles))+
			") does not equal the length of the angle_weights ("
			+str(len(angle_weights)) )
		
		self.Na = len(angles)
		self.angles = np.asarray(angles)
		self.angle_weights = np.asarray(angle_weights)
		
	

def angle_information_with_natural_weights (angles,geometry):
	"""
	Given a set of angles, creates an angle_information object
	with 
	"""

	if geometry == gratopy.RADON:
		full_interval =  np.pi
	elif geometry == gratopy.FANBEAM:
		full_interval = 2 * np.pi
	else:
		raise Exception("The \"geometry\" argument given to " + 
		"\"angle_information_with_natural_weights\" has to be " +
		"gratopy.RADON(=1) or gratopy.FANBEAM(=2). Given was " + 
		str(geometry)) 
	angles = np.asarray(angles)
	if geometry == gratopy.RADON:
		angles_index = np.argsort(angles % (full_interval))
		angles_sorted = angles[angles_index] % (full_interval)
		angles_extended = np.array(
		np.hstack([-full_interval + angles_sorted[-1], angles_sorted,
			angles_sorted[0] + full_interval]))
		na = len(angles_sorted)
		angle_weights = 0.5 * (abs(angles_extended[2 : na + 2] - angles_extended[0:na]))



		# Correct for multiple occurrence of angles, for example
		# angles in [0,2pi] are considered instead of [0,pi]
		# and mod pi has same value) The weight of the same angles
		# is distributed equally.
		tol = 0.000001
		na = len(angles_sorted)
		i = 0
		while i < na - 1:
			count = 1
			my_sum = angle_weights[i]
			while abs(angles_sorted[i] - angles_sorted[i + count]) < tol:
				my_sum += angle_weights[i + count]
				count += 1
				if i + count > na - 1:
					break

			val = my_sum / count
			for j in range(i, i + count):
				angle_weights[j] = val
			i += count

		angle_weights[angles_index] = angle_weights
		return angle_information(angles,angle_weights)




if __name__=="__main__":
	#my_angles = uniform_angles(10,2)
	#my_angles = uniform_angles_in_interval(10,0 , np.pi/2)
	#my_angles = uniform_angles_on_many_intervals([10,5] , [0,np.pi/2] , [np.pi/2,np.pi])
	#my_angles = angle_information(np.array([0,5]) , [1,2])
	#my_angles = angle_information_with_natural_weights([0,1,1,1,2,2,1+np.pi],gratopy.RADON)
	#my_angles = angle_information_with_one_weights([0,1,3,2,1])
	print (my_angles.Na)
	print (my_angles.angles) 
	print (my_angles.angle_weights)
