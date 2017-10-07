import os
import sys
import unittest
from datetime import datetime

# add sibling folder "src" to the syspath --> important note: cannot use relative path like "../src"
# because then running the tests from different directories would screw it up

# http://docs.python-guide.org/en/latest/writing/tests/
# testing basics

sys.path.insert(0,os.path.abspath(
	os.path.join(os.path.join(
	os.path.abspath(os.path.join(os.path.dirname(
	os.path.realpath(__file__)), os.pardir)),"./src"))))

import timerep

class TestCron(unittest.TestCase):
 
	def setUp(self):
		pass
 
	def test_staticdates(self): #DOKONCIT
		d1 = datetime(2012,3,9,0)
		d2 = datetime(2012,3,17,19,59)
		cronexprs = ["* * * * *"]
		for expr in cronexprs:
			cron = timerep.Cron(expr)
			self.assertTrue(cron.evaluate(d1))
			self.assertTrue(cron.evaluate(d2))
 
 
if __name__ == '__main__':
	unittest.main()
