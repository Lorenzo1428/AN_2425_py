import numpy as np
import metodi_numerici as m
f = m.fattorizzazioni()
A = 2
A,B = f.lu_(A)


print(A,B)
