import numpy as np
import argparse
from configs import *

class CliffWorld:
	def __init__(self,gridRow=4,gridCol=12,rCliff=-100,rNormal=-1):
		self.gridRow = gridRow
		self.gridCol = gridCol
		self.gridWorld = np.zeros([gridRow,gridCol],dtype=bool)
		# set the cliff by default
		self.gridWorld[gridRow-1,1:(gridCol-1)] = True # it's cliff
		self.start = [gridRow-1,0]
		self.goal = [gridRow-1,gridCol-1]
		self.gamma = 1
		
		self.rCliff = rCliff
		self.rNormal = rNormal
		
		self.actionNum = ACTION_NUM # policy from configs
		self.dirx = DIR_X
		self.diry = DIR_Y
		
		self.actionValue = np.zeros([ACTION_NUM,gridRow,gridCol],dtype=np.float64)

	def printValues(self):
		print(self.actionValue)

	def printPolicies(self):
		policies = ["" for _ in range(0,self.gridRow*self.gridCol)]
		for i in range(0,self.gridRow):
			for j in range(0,self.gridCol):
				a = np.argmax(self.actionValue[:,i,j])
				#print(self.actionValue[i,j,:])
				policies[i*self.gridRow+j] += ACTIONS[a]
		formated = ""
		for i in range(0,self.gridRow):
			for j in range(0,self.gridCol):
				formated+="| {} |".center(6)
			formated+='\n'
		print(formated.format(*policies))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gridRow', type=int, default=2, help="the gridRow size")
	parser.add_argument('--gridCol', type=int, default=3, help="the gridCol size")
	parser.add_argument('--rCliff', type=float, default=-
						100, help="the future reward for Cliff grids")
	parser.add_argument('--rNormal', type=float, default=-
						1, help="the future reward for normal grids")
	args = parser.parse_args()
	gridRow = args.gridRow
	gridCol = args.gridCol
	rCliff = args.rCliff
	rNormal = args.rNormal

	cliffWorld = CliffWorld(gridRow,gridCol,rCliff,rNormal)
	cliffWorld.printPolicies()
	cliffWorld.printValues()
