from generate import *
from dfa import *

true_exp = float(input('Insert true scaling exponent: '))
x = power_law_noise(2**12, true_exp)
scales, fluct, alpha = dfa(x, show = 1)
print("True scaling exponent: {}".format(true_exp))
print("Estimated DFA exponent: {}".format(alpha))


x = power_law_noise(2**12, true_exp)
scales, fluct, alpha = dfawithoverlap(x, show = 1, overlap = 50)
print("True scaling exponent: {}".format(true_exp))
print("Estimated DFA exponent: {}".format(alpha))