import os
import sys

# add sibling folder "src" to the syspath --> important note: cannot use relative path like "../src"
# because then running the tests from different directories would screw it up

sys.path.insert(0,os.path.abspath(
	os.path.join(os.path.join(
	os.path.abspath(os.path.join(os.path.dirname(
	os.path.realpath(__file__)), os.pardir)),"./src"))))


import timerep
import unittest
from datetime import datetime

class TestCron(unittest.TestCase):
 
	def setUp(self):
		self.cron = timerep.Cron("* * * * *")

	def test_numbers_3_4(self):
		self.assertEqual(self.cron.evaluate(datetime(2012,3,1,1,1)),True)
 
 
if __name__ == '__main__':
	# print ("Printing",os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
	# print ("Pardir", os.pardir)
	# print ("hhh",os.path.dirname(os.path.realpath(__file__)))
	# h = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
	# print ("stvrte",os.path.abspath(os.path.join(h,"./src")))
	# print ("piate", os.path.abspath("../src"))
	print (os.path.abspath(
	os.path.join(os.path.join(
	os.path.abspath(os.path.join(os.path.dirname(
	os.path.realpath(__file__)), os.pardir)),"./src"))))
	unittest.main()
